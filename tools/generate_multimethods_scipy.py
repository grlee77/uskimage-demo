import inspect
import os
import pickle
import re
import types
from textwrap import dedent

import black

from scipy import linalg, special

# import isort


modules = [linalg, special]

# cp __init__.py _api.py


preamble_1 = """
import functools
import warnings

import numpy as np
from numpy import dtype, ndarray
from uarray import (Dispatchable, all_of_type, create_multimethod,
                    generate_multimethod)
from unumpy import mark_dtype

from . import _api
"""

preamble_2 = f'''

_mark_output = functools.partial(ua.Dispatchable,
                                 dispatch_type=image_output,
                                 coercible=False)


_mark_scalar_or_array = functools.partial(
    ua.Dispatchable, dispatch_type=scalar_or_array, coercible=True
)

def _get_docs(func):
    """
    Decorator to take the docstring from original
    function and assign to the multimethod.
    """
    func.__doc__ = getattr(_api, func.__name__).__doc__
    return func


def _identity_replacer(args, kwargs, arrays):
    return args, kwargs

'''


def get_dispatchables(sig, dispatchable_names):
    parameter_names = tuple(sig.parameters.keys())
    dispatchables = []
    dispatchable_params = []
    for p in parameter_names:
        dispatch_type = dispatchable_names.get(p, None)
        if dispatch_type:
            dispatchable_params.append(p)
            if dispatch_type == 'ndarray':
                dispatchables.append(p)
            elif dispatch_type == 'dtype':
                dispatchables.append(f'Dispatchable({p}, np.dtype)')
            elif dispatch_type == 'noncoercable_ndarray':
                dispatchables.append(f'_mark_non_coercible({p})')
    return dispatchables, dispatchable_params


def _fix_function_params(param_str):
    while '<function ' in param_str:
        fstart = param_str.find('<function')
        fend = fstart + param_str[fstart:].find('>')
        func_name = param_str[fstart:fend].split(' ')[1]
        param_str = param_str[:fstart] + func_name + param_str[fend + 1:]
    while '<lambda ' in param_str:
        fstart = param_str.find('<lambda')
        fend = fstart + param_str[fstart:].find('>')
        param_str = param_str[:fstart] + 'lambda x: None' + param_str[fend + 1:]
    while 'dtype=<class ' in param_str:
        fstart = param_str.find('dtype=<class ') + 6
        fend = fstart + param_str[fstart:].find('>')
        dparam = param_str[fstart:fend].split(" ")[1].replace('numpy.', '')
        param_str = param_str[:fstart] + dparam + param_str[fend + 1:]
    return param_str


def _get_partial_signature(sig, dispatchables, dispatchable_params):
    if not dispatchables:
        return [], [], {}, {}
    last_dispatchable = dispatchables[-1]
    args = []
    modified_args = []
    kwargs = {}
    modified_kwargs = {}
    cnt = 0
    any_kw_only = False
    for name in sig.parameters.keys():
        p = sig.parameters[name]
        param_str = str(p)
        param_str = _fix_function_params(param_str)
        if p.kind == p.KEYWORD_ONLY and not any_kw_only:
            modified_kwargs['kwonly'] = '*'
            kwargs['kwonly'] = '*'
        if (p.kind == p.POSITIONAL_ONLY or (p.kind == p.POSITIONAL_OR_KEYWORD
                                  and p.default == inspect.Parameter.empty)):
            args.append(param_str)
            if p.name in dispatchable_params:
                modified_args.append(f'dispatchables[{cnt}]')
                cnt += 1
            else:
                modified_args.append(param_str)
        else:
            if param_str in ["*args", "**kwargs"]:
                continue
            k, v = param_str.split('=')
            kwargs[k] = v
            if p.name in dispatchable_params:
                modified_kwargs[k] = f'dispatchables[{cnt}]'
                cnt += 1
        if name == dispatchable_params[-1]:
            break

    return args, kwargs, modified_args, modified_kwargs


def _name_from_args(args, kwargs):
    replacer_name = '_' + '_'.join(n.replace('_', '') for n in args)
    if kwargs:
        replacer_name += '_kw_' + '_'.join(k.replace('_', '') for k in kwargs.keys())
    replacer_name += '_replacer'
    replacer_name = replacer_name.replace('kw_kwonly', 'kwonly')
    return replacer_name


def get_replacer(sig, dispatchables, dispatchable_params):
    if dispatchables == []:
        replacer_name = '_identity_replacer'
    elif dispatchables == ['image']:
        replacer_name = '_image_replacer'
    else:
        args, kwargs, modified_args, modified_kwargs = _get_partial_signature(sig, dispatchables, dispatchable_params)
        replacer_name = _name_from_args(args, kwargs)
    return replacer_name


def generate_replacer_code(sig, dispatchables, dispatchable_params):
    args, kwargs, modified_args, modified_kwargs = _get_partial_signature(sig, dispatchables, dispatchable_params)
    replacer_name = _name_from_args(args, kwargs)

    any_kw_only = 'kwonly' in kwargs
    if (len(args) + len(kwargs)) == 0:
        inner_arg_list = '*args, **kwargs'
    else:
        inner_arg_list = ', '.join(args)
        if any_kw_only:
            keys = list(kwargs.keys())
            n_kwargs_to_list = keys.index('kwonly')
            # subset of kwargs that are not keyword-only
            kwargs = {k: kwargs[k] for k in keys[:n_kwargs_to_list]}
            # dict only containing dispatchable keyword-only args
            mkeys = list(modified_kwargs.keys())
            n_kwdisp_nonkwonly = mkeys.index('kwonly')
            kwonly_args = {k: modified_kwargs[k] for k in mkeys[n_kwdisp_nonkwonly + 1:]}
        else:
            kwonly_args = {}
            n_kwdisp_nonkwonly = len(modified_kwargs)
            n_kwargs_to_list = len(kwargs)
        if n_kwargs_to_list:
            kw_list = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
            # TODO: remove ={v} as in the commented line below?
            # kw_list = ', '.join(list(kwargs.keys())[:n_kwargs_to_list])
            if inner_arg_list:
                inner_arg_list += f', {kw_list}'
            else:
                inner_arg_list = f'{kw_list}'
        if not any_kw_only:
            inner_arg_list += ', *args'
        inner_arg_list += ', **kwargs'

    if modified_kwargs:
        kw_copy = 'kw_out = kwargs.copy()'
    else:
        kw_copy = 'kw_out = kwargs'

    # cnt is the number of dispatchables within modified_args
    cnt = len([m for m in modified_args if m.startswith('dispatchable')])
    for k, v in kwargs.items():
        if k in modified_kwargs:
            # list dispatchable item in place of name
            modified_args.append(f'dispatchables[{cnt}]')
            cnt += 1
        else:
            # append name of non-dispatchable
            modified_args.append(k)

    kw_replacements = ""
    any_kw_only = False
    for i, (k, v) in enumerate(kwonly_args.items()):
        indent = 0 if i == 0 else 16  # i == 1 due to kwonly
        kw_replacements += " " * indent + f"if '{k}' in kw_out:"
        kw_replacements += " " * 20 + f"kw_out['{k}'] = dispatchables[{i + cnt}]\n"

    if len(modified_args) > 1:
        disp_args_tuple = '(' + ', '.join(modified_args) + ')'
    elif len(modified_args) == 1:
        disp_args_tuple = f'({modified_args[0]},)'
    else:
        disp_args_tuple = '()'
    if '*args' in inner_arg_list:
        # all possible positional args not listed, so add + args
        disp_args_tuple = disp_args_tuple + '+ args'

    replacer_code = dedent(f'''
        def {replacer_name}(args, kwargs, dispatchables):
            def self_method({inner_arg_list}):
                {kw_copy}
                {kw_replacements}
                return {disp_args_tuple}, kw_out
            return self_method(*args, **kwargs)
    ''')

    # remove any blank lines
    replacer_code = '\n'.join([r for r in replacer_code.split('\n') if r.strip()])
    return replacer_code

"""

if False:
    # k = 'flood_fill'
    # k = 'remove_small_objects'
    k = 'h_minima'
    v = mod.__dict__[k]
    sig = inspect.signature(v)
    dispatchables, dispatchable_params = get_dispatchables(sig)
    args, kwargs, modified_args, modified_kwargs = _get_partial_signature(sig, dispatchables, dispatchable_params)
    replacer_name = get_replacer(sig, dispatchables, dispatchable_params)
    replacer_code = generate_replacer_code(sig, dispatchables, dispatchable_params)
"""

def main(mod, out_file='/tmp/mm.py', dispatchable_dict={}):
    all_replacers = {}
    replacer_codes = []
    multimethods = ""

    # If module has already had __init__ modified to use multimethods then we
    # need to instead get the original functions from the _api file.
    api_mod = mod.__dict__.get('_api', mod)

    items = sorted(api_mod.__dict__.items(), key=lambda x: x[0])
    for k, v in items:
        if isinstance(v, types.FunctionType):
            sig = inspect.signature(v)
            dispatchables, dispatchable_params = get_dispatchables(sig, dispatchable_names=dispatchable_dict)
            if len(dispatchables) > 1:
                dispatchables_tuple = '(' + ', '.join(dispatchables) + ')'
            elif len(dispatchables) == 1:
                dispatchables_tuple = f'({dispatchables[0]},)'
            else:
                dispatchables_tuple = '()'

            replacer_name = get_replacer(sig, dispatchables, dispatchable_params)

            sig_str = str(sig)
            sig_str = _fix_function_params(sig_str)

            print(f"name={k}, replacer_name={replacer_name}")
            multimethods += dedent(f'''


            @create_skimage_{mod.__name__.split('.')[-1]}({replacer_name})
            @all_of_type(ndarray)
            @_get_docs
            def {v.__name__ + sig_str}:
                return {dispatchables_tuple}
            ''')

            if replacer_name not in ['_identity_replacer', '_image_replacer']:
                if replacer_name not in all_replacers:
                    replacer_code = generate_replacer_code(sig, dispatchables, dispatchable_params)
                    replacer_codes.append(replacer_code)
                    all_replacers[replacer_name] = replacer_code
                    try:
                        replacer_code = black.format_str(
                            replacer_code,
                            mode=black.Mode(
                                target_versions={black.TargetVersion.PY36},
                                line_length=79,
                                string_normalization=False,
                                is_pyi=False,
                            )
                        )
                    except black.parsing.InvalidInput:
                        print(f"BROKEN replacer_code={replacer_code}")


    mm_file = preamble_1
    mm_file += dedent(f'''
    __all__ = {sorted(mod.__all__)}

    create_skimage_{mod.__name__.split('.')[-1]} = functools.partial(
        create_multimethod, domain="numpy.{mod.__name__}"
    )
    ''')


    mm_file += preamble_2

    for key in all_replacers.keys():
        mm_file += "\n\n" + all_replacers[key]

    mm_file += multimethods

    mm_file = black.format_str(
        mm_file,
        mode=black.Mode(
            target_versions={black.TargetVersion.PY36},
            line_length=79,
            string_normalization=False,
            is_pyi=False,
        )
    )
    # mm_file = isort.code(mm_file)
    with open(out_file, 'wt') as f:
        f.write(mm_file)


def _get_dispatchables_dict(mod):
    items = sorted(mod.__dict__.items(), key=lambda x: x[0])
    ndarray_names = set()
    # note: Could refactor to use numpydoc to iterate over parameter
    #       descriptions.
    for k, v in items:
        if isinstance(v, types.FunctionType):
            doc = v.__doc__
            if doc is None or 'Parameters' not in doc:
                continue
            pdocs = doc.split('Parameters')[1].split('Returns')[0];
            array_params = [line.strip() for line in pdocs.split('\n')
                            if ': ' in line and ('array' in line or 'matrix' in line)]
            array_param_names = [p.split(':')[0].strip() for p in array_params]
            ndarray_names = ndarray_names | set(array_param_names)
    d = {k: 'ndarray' for k in sorted(list(ndarray_names))}
    # Fix some particular ndarray cases where automatic docstring parsing as
    # above does not work. e.g. doesn't handle multiple grouped parameters like
    #     im1, im2 : ndarray
    # d['image'] = 'ndarray'

    # manually curated list of dispatchables that are not purely ndarray
    d['dtype'] = 'dtype'
    # d['out'] = 'noncoercable_ndarray'

    return d


if __name__ == '__main__':

    # loop over scikit-image modules, writing _multimethods_NAME.py out to /tmp
    for mod in modules:
        out_name = f'_multimethods_{mod.__name__.split(".")[-1]}.py'
        disp_names = _get_dispatchables_dict(mod)
        # print(f"disp_names={disp_names}")
        main(mod, out_name, dispatchable_dict=disp_names)

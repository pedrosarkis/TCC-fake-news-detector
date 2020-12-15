import os
from functools import wraps
from flask import request


def auth():
    def decorator(f):
        @wraps(f)
        def decorate(*args, **kwargs):
            token = request.headers.get('Authorization')
            if token != os.environ.get('TOKEN', ''):
                return {'status': 'error', 'cause': 'invalid_token'}, 403
            return f(*args, **kwargs)
        return decorate
    return decorator


def body(parser_class=None):
    def decorator(f):
        @wraps(f)
        def decorate(*args, **kwargs):
            if parser_class is not None:
                try:
                    kwargs['body'] = parser_class.parser().parse_args()
                except Exception as e:
                    return {'status': 'error', 'cause': 'invalid_body', 'payload': e.data}, 400
            else:
                kwargs['body'] = request.json or {}

            return f(*args, **kwargs)
        return decorate
    return decorator

from aiohttp import web

from .views import Recognition

routes = [
    web.view('/recognition', Recognition)
]
import argparse
import logging

from aiohttp import web
from face_segmentation.hair_segmentation import prepare_model
from ms_utils.logger import AccessLogger
from ms_utils.routes import routes


def run_server(host, port):
    prepare_model('./')

    app = web.Application()
    app.add_routes(routes=routes)
    web.run_app(app, host=host, port=port, access_log_class=AccessLogger)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="aiohttp server")
    parser.add_argument('--host', required=False, default="192.168.1.108")
    parser.add_argument('--port', required=False, default="60604")

    args = parser.parse_args()
    logging.info("app.py: Server is about to start.")
    run_server(host=args.host, port=args.port)

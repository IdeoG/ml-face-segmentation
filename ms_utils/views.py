import base64
import io

from PIL import Image
from aiohttp import web


class RecognitionPresenter:

    def __init__(self):
        self._model = None

    async def recognize_face(self, data):
        bytes = base64.b64decode(data['base64image'])
        image = Image.open(io.BytesIO(bytes))
        _ = self._model.embeddings(image)

        response = {}
        return response


class BaseView(web.View):

    def __init__(self, request):
        super().__init__(request)

        self._presenter = RecognitionPresenter()


class Recognition(BaseView):
    async def post(self):
        data = await self.request.json()

        response = await self._presenter.recognize_face(data)
        return web.json_response(response)

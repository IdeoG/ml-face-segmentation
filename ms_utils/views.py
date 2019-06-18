import base64
import io

from PIL import Image
from aiohttp import web

from face_segmentation.hair_segmentation import inference


class RecognitionPresenter:

    async def segmentation(self, data):
        bytes = base64.b64decode(data['base64image'])
        image = Image.open(io.BytesIO(bytes))
        mask = inference(image)

        response = {'mask': base64.b64encode(mask.tobytes())}
        return response


class BaseView(web.View):

    def __init__(self, request):
        super().__init__(request)

        self._presenter = RecognitionPresenter()


class Recognition(BaseView):
    async def post(self):
        data = await self.request.json()

        response = await self._presenter.segmentation(data)
        return web.json_response(response)

from aiohttp.abc import AbstractAccessLogger


class AccessLogger(AbstractAccessLogger):

    def log(self, request, response, time):
        info = "\nHTTP client --> server request info:\n" \
               "\tremote = {0}\n" \
               "\tmethod = {1}\n" \
               "\tpath = {2}\n" \
               "\tcompleted in {3}ms\n" \
               "HTTP client <-- server response info:\n" \
               "\tstatus = {4}\n" \
               "\tbody = {5}\n"\
            .format(request.remote, request.method, request.path, int(time*1000), response.status, response.body)
        self.logger.info(info)
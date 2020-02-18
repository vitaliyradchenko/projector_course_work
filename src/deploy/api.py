import tornado

from ..predictor import NewsPredictor


class NewsClassifier(tornado.web.RequestHandler):
    def initialize(self, predictor: NewsPredictor, version: int) -> None:
        self.predictor = predictor
        self.version = version

    async def post(self) -> None:
        samples = tornado.escape.json_decode(self.request.body)["samples"]
        predictions = self.predictor.predict(samples)
        result = {"version": self.version, "results": predictions}
        self.write(result)
        self.set_status(200)


class Healthcheck(tornado.web.RequestHandler):
    def initialize(self, version: int) -> None:
        self.version = version

    async def get(self) -> None:
        self.write({"results": "healthy", "service": "sentiment", "version": self.version})
        self.set_status(200)

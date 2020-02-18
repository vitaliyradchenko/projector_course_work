import logging

import tornado.ioloop
import tornado.web
from tornado.httpclient import AsyncHTTPClient

from src.deploy.api import Healthcheck, NewsClassifier
from src.predictor import NewsPredictor

logging.getLogger("transformers").setLevel(logging.ERROR)


MODEL_PATH = "model/"
PORT = 1492


def make_app(predictor: NewsPredictor, version: int) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/news_classifier/healthcheck", Healthcheck, dict(version=version)),
            (r"/news_classifier/", NewsClassifier, dict(predictor=predictor, version=version)),
        ]
    )


if __name__ == "__main__":
    AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient", max_clients=100)
    http_client = AsyncHTTPClient()
    model_version = int(open("model.version").readline().rstrip())
    news_predictor = NewsPredictor.load(MODEL_PATH)
    app = make_app(predictor=news_predictor, version=model_version)
    app.listen(PORT)
    tornado.ioloop.IOLoop.current().start()

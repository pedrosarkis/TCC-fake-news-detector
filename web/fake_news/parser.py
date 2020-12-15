from flask_restful import reqparse


class PredictParser:
    @staticmethod
    def parser():
        parser = reqparse.RequestParser(bundle_errors=True)

        # Primeira etapa
        parser.add_argument('content', type=str, required=True, help='content is required')
        return parser

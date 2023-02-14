from flask import Flask, request, jsonify
import recommendation_system as rs
import time
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    try:
        customer_id = request.args.get("customer_id")
        date = request.args.get("date")

        if customer_id is None or date is None:
            return "Error: customer_id and date parameters are required", 400

        # Simulate a delay for loading recommendation
        #time.sleep(15)
        recom_engine = rs.RecommendationEngine(root = 'cleaned_online_retail.csv')
        recommendations = recom_engine.recommend_products(float(customer_id),date_str=str(date))

        print(jsonify(recommendations.tolist()))
        # Show the recommendations once they have loaded
        return jsonify(recommendations.tolist())
    except ValueError as e:
        return jsonify(str(e)), 400;


if __name__ == "__main__":
    app.debug = True
    app.run()

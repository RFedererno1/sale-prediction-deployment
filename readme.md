# Codebase Overview
====================

## Structure
Our codebase is organized into the following folders:

* `app`: Contains the main application code
    + `db/schema.py`: API responses models
	+ `routes`: Ruote for 2 backend APIs.
        - `inference.py`: Get user's request: item_id, push it to MQ and return pending response.
        - `result.py`: The result of item_id prediction: get result from Redis, if data is still processing, return pending response.
	+ `app.py`: Backend routes.
    + `main.py`: main file to start the API service.
* `consumer/inference.py`: get the item_id in MQ, predict and update the result to Redis.
* `data`: Contains the data used by the application
* `model`: contain the prediction model: load model, preprocessing, inference.
* `rabbitmq_config`: connect rabbitMQ and create queue.
* `redis_config`: connect Redis.
* `scripts/sale_prediction_script.ipynb`: the jupyter notebook file containing the modeling code for Task 1.
* `weight`: contains the weight file for the prediction model.
* `.env`: Redis and RabbitMQ connection.
* `UI.py`: Sales Prediction App for internal user interaction.
* `docker files`: Docker configuration files for deployments.

## Running the Code
To run the code, follow these steps:

1. Navigate to the root directory of the codebase
2. Run the command `docker-compose up --build` to start the Docker containers
3. The UI will be exposed on port `8080` (or the port specified in the `docker-compose.yml` file)
4. Wait until the line: "INFO:    Done loading model" appears in the terminal.
5. Access the UI by navigating to `http://localhost:8080` in your web browser and start for usage.

Note:
- Make sure you have Docker and Docker Compose installed on your system before running the code.
- Make sure that there is no application running on 8000 and 8080 port.

# Modeling
====================
My modeling jupyter notebook file is in `scripts/sale_prediction_script.ipynb`

# Some deployment patterns fit to this situation, and my choice to implement.
====================

Several deployment patterns can suit this situation. Since it is for internal users, we can prioritize simplicity and ease of development.

## Backend
Given the limited number of internal users, handling a high volume of requests isn't a concern. Therefore, FastAPI is an excellent choice for the backend. It's lightweight, easy to code, and facilitates fast deployment. Additionally, it can handle a reasonable number of parallel requests, making it suitable for our needs.

## Frontend
The interface only needs to display results without requiring complex UI or navigation. Streamlit is a great option here. It's designed for rapid development of simple user interfaces and is perfect for proof-of-concept (POC) applications. With built-in components, we can build a functional UI in under 100 lines of code.

## Task Processing
Machine learning tasks can be time-consuming and are CPU-bound. To manage this, a message queue is necessary to store queries, ensuring that none are lost during long processing times. RabbitMQ is ideal for this due to its simple setup and efficient processing. It ensures no query is missed while handling long-running tasks.

This setup provides a balanced solution with fast development and reliable performance for internal use.

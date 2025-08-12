from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from datetime import datetime
import pandas as pd

# Import the classes that need to be serialized
from livetrader import LiveTrader, Budget
from strategies.pre_trade import PreTrade
from strategies.active_trade import ActiveTrade
from api_executor import ApiExecutor  # Import to identify and exclude it


class CustomJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle the serialization of the LiveTrader object
    and its nested components.
    """

    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, (Budget, PreTrade, ActiveTrade)):
            # For our custom classes, we serialize their __dict__ attribute
            return obj.__dict__
        if isinstance(obj, pd.DataFrame):
            # For DataFrames, convert them to a dictionary
            return obj.to_dict(orient='records')
        # Let the base class default method raise the TypeError for unhandled types
        return json.JSONEncoder.default(self, obj)


class StatusServer:
    """
    A class that runs an HTTP server in a separate thread to provide
    a full JSON representation of a LiveTrader instance.
    """

    def __init__(self, live_trader: LiveTrader, host='localhost', port=8000):
        self.live_trader = live_trader
        self.host = host
        self.port = port

        # Create a custom request handler that has access to the live_trader instance
        class RequestHandler(BaseHTTPRequestHandler):
            trader = self.live_trader

            def do_GET(self):
                if self.path == '/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()

                    try:
                        # Create a copy of the trader's dictionary to avoid modifying the original object
                        trader_dict = self.trader.__dict__.copy()

                        # --- Key Change: Exclude the non-serializable ApiExecutor ---
                        if 'api_executor' in trader_dict:
                            del trader_dict['api_executor']

                        if 'logger' in trader_dict:
                            del trader_dict['logger']

                        if 'MAX_CYCLE_DURATION' in trader_dict:
                            del trader_dict['MAX_CYCLE_DURATION']

                        if 'REINVESTMENT_WINDOW' in trader_dict:
                            del trader_dict['REINVESTMENT_WINDOW']

                        # Serialize the modified dictionary using our custom encoder
                        json_output = json.dumps(trader_dict, cls=CustomJSONEncoder, indent=4)
                        self.wfile.write(json_output.encode('utf-8'))
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        error_payload = json.dumps(
                            {'error': 'Failed to serialize LiveTrader object', 'details': str(e)})
                        self.wfile.write(error_payload.encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Not Found. Use /status endpoint."}')

        self.server = HTTPServer((self.host, self.port), RequestHandler)

    def start(self):
        """Starts the HTTP server in a new thread."""
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print(f"Status server started on http://{self.host}:{self.port}")

    def stop(self):
        """Stops the HTTP server."""
        self.server.shutdown()
        self.server.server_close()
        print("Status server stopped.")


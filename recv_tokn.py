import asyncio
import websockets
import json
import os
from datetime import datetime

async def subscribe_public_feed():
    uri = "wss://futures.kraken.com/ws/v2"
    ticker = "PI_XBTUSD"
    
    # Create data directory
    data_folder = f"{ticker}_DATA"
    os.makedirs(data_folder, exist_ok=True)

    subscription_message = {
        "event": "subscribe",
        "feed": "ticker",  
        "product_ids": [ticker]
    }

    async with websockets.connect(uri) as ws:
        # Send subscription request
        await ws.send(json.dumps(subscription_message))
        print(f"Subscribed to ticker feed for {ticker}...")

        # Continuously receive messages
        while True:
            try:
                raw_msg = await ws.recv()  # receive raw message
                msg = json.loads(raw_msg)  # parse JSON
                
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(data_folder, f"ticker_data_{timestamp}.json")
                
                # Save message to JSON file
                with open(filename, 'w') as f:
                    json.dump(msg, f, indent=4)
                
                print(f"Saved data to {filename}")
                print(msg)
                
            except websockets.ConnectionClosed:
                print("Connection closed.")
                break

if __name__ == "__main__":
    asyncio.run(subscribe_public_feed())

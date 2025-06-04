import os
from pymongo import MongoClient
import urllib.parse

def test_mongodb_connection():
    # Original password
    password = "Jasmine@0802"
    # URL encode the password
    encoded_password = urllib.parse.quote_plus(password)
    
    # Construct the URI with encoded password
    mongo_uri = f"mongodb+srv://jaschri:{encoded_password}@agent.wb3vq0q.mongodb.net/?retryWrites=true&w=majority&appName=Agent"
    
    try:
        # Try to connect
        client = MongoClient(mongo_uri)
        # Test the connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        return True
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb_connection() 
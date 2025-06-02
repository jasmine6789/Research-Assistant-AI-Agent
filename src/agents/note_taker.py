from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING
import datetime
import json

class NoteTaker:
    def __init__(self, mongo_uri: str, db_name: str = "arxiv_logs", collection_name: str = "logs"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        # Performance: Add indexes for common queries
        self.collection.create_index([("type", ASCENDING)])
        self.collection.create_index([("user", ASCENDING)])
        self.collection.create_index([("session_id", ASCENDING)])
        self.collection.create_index([("timestamp", ASCENDING)])

    def log(self, log_type: str, content: Dict[str, Any], user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None, provenance: Optional[Dict[str, Any]] = None):
        """
        Log an event with type, content, timestamp, user, session_id, agent, and provenance.
        """
        entry = {
            "type": log_type,
            "content": content,
            "timestamp": datetime.datetime.utcnow(),
        }
        if user:
            entry["user"] = user
        if session_id:
            entry["session_id"] = session_id
        if agent:
            entry["agent"] = agent
        if provenance:
            entry["provenance"] = provenance
        self.collection.insert_one(entry)

    # --- User and Session Management ---
    def log_session_start(self, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("session_start", {}, user, session_id, agent)

    def log_session_end(self, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("session_end", {}, user, session_id, agent)

    def get_sessions(self, user: Optional[str] = None) -> List[str]:
        """Return a list of session_ids for a user (or all users)."""
        query = {"type": "session_start"}
        if user:
            query["user"] = user
        return [s["session_id"] for s in self.collection.find(query, {"session_id": 1, "_id": 0}) if "session_id" in s]

    # --- Log Retention and Archiving ---
    def delete_old_logs(self, days: int = 30) -> int:
        """Delete logs older than the specified number of days. Returns count deleted."""
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        result = self.collection.delete_many({"timestamp": {"$lt": cutoff}})
        return result.deleted_count

    def archive_old_logs(self, days: int = 30, file_path: str = "archived_logs.json") -> int:
        """Export logs older than the specified number of days to a JSON file and delete them. Returns count archived."""
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        old_logs = list(self.collection.find({"timestamp": {"$lt": cutoff}}, {"_id": 0}))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(old_logs, f, default=str, indent=2)
        result = self.collection.delete_many({"timestamp": {"$lt": cutoff}})
        return result.deleted_count

    # --- Audit Trail and Provenance ---
    def edit_log(self, log_id: Any, new_content: Dict[str, Any], user: Optional[str] = None, agent: Optional[str] = None):
        """Edit a log entry, storing previous version in provenance."""
        old_log = self.collection.find_one({"_id": log_id})
        if not old_log:
            return False
        provenance = old_log.get("provenance", [])
        provenance.append({
            "content": old_log["content"],
            "edited_by": user,
            "edited_at": datetime.datetime.utcnow(),
            "agent": agent
        })
        self.collection.update_one({"_id": log_id}, {"$set": {"content": new_content, "provenance": provenance}})
        return True

    def delete_log(self, log_id: Any, user: Optional[str] = None, agent: Optional[str] = None):
        """Delete a log entry, storing a deletion record in provenance."""
        old_log = self.collection.find_one({"_id": log_id})
        if not old_log:
            return False
        provenance = old_log.get("provenance", [])
        provenance.append({
            "deleted_by": user,
            "deleted_at": datetime.datetime.utcnow(),
            "agent": agent
        })
        self.collection.update_one({"_id": log_id}, {"$set": {"provenance": provenance}})
        self.collection.delete_one({"_id": log_id})
        return True

    # --- Existing logging methods (unchanged, but now support agent/provenance) ---
    def log_query(self, query: str, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("query", {"query": query}, user, session_id, agent)

    def log_selected_papers(self, paper_ids: List[str], user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("selected_papers", {"paper_ids": paper_ids}, user, session_id, agent)

    def log_hypothesis(self, hypothesis: str, refined: bool = False, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("hypothesis", {"hypothesis": hypothesis, "refined": refined}, user, session_id, agent)

    def log_feedback(self, feedback: str, reason: Optional[str] = None, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("feedback", {"feedback": feedback, "reason": reason}, user, session_id, agent)

    def log_insight(self, insight: str, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("insight", {"insight": insight}, user, session_id, agent)

    def log_visualization(self, viz_type: str, customization: Dict[str, Any], user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("visualization", {"viz_type": viz_type, "customization": customization}, user, session_id, agent)

    def log_regeneration(self, stage: str, reason: str, user: Optional[str] = None, session_id: Optional[str] = None, agent: Optional[str] = None):
        self.log("regeneration", {"stage": stage, "reason": reason}, user, session_id, agent)

    # --- Retrieval (unchanged) ---
    def get_logs(self, log_type: Optional[str] = None, user: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve logs filtered by type, user, or session_id.
        """
        query = {}
        if log_type:
            query["type"] = log_type
        if user:
            query["user"] = user
        if session_id:
            query["session_id"] = session_id
        return list(self.collection.find(query, {"_id": 0})) 
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from rag_helper import answer_question


class ActionHealthBotRAG(Action):

    def name(self) -> Text:
        return "action_healthbot_rag"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict
    ) -> List[Dict]:

        try:
            user_msg = tracker.latest_message.get("text", "")
            lang = tracker.get_slot("language") or "en"

            reply = answer_question(user_msg, lang)

            if not reply:
                reply = "Sorry — I couldn't find reliable information."

            dispatcher.utter_message(text=reply)

        except Exception as e:
            dispatcher.utter_message(text=f"Internal error: {e}")

        return []


class ActionSetLanguage(Action):

    def name(self):
        return "action_set_language"

    def run(self, dispatcher, tracker, domain):
        lang = tracker.latest_message.get("text", "en")
        dispatcher.utter_message(text=f"Language set to {lang}")
        return [SlotSet("language", lang)]

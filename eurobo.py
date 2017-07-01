# -*- coding: utf-8 -*-

import argparse
from threading import Thread
from queue import Queue
import sys, os, time
import json, requests
import webbrowser

# training model
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

# training_data = load_data('data/_eurobo.json')
# trainer = (Trainer(RasaNLUConfig("_config_spacy_eurobo.json")))
# trainer.train(training_data)
# model_directory = trainer.persist('./model/')

# prediction
from rasa_nlu.model import Metadata, Interpreter

metadata = Metadata.load("./model/model_20170629-120440/")   # where model_directory points to the folder the model is persisted in
interpreter = Interpreter.load(metadata, RasaNLUConfig("_config_spacy_eurobo.json"))

# import seeing

#url = "http://localhost:5000/parse?q="

last_intent = ""
terms_to_search = ""

def process_input_sentence(sentence):
	answer = ""
	print(last_intent)

	# response = requests.get(url + sentence)
	# response.raise_for_status()
	r = interpreter.parse(sentence)
	# print("response: ", r)

	
	intent = r.get("intent").get("name")
	entities = r.get("entities")

	# print("intent: "+ intent + "\n")
	if intent == "goodbye":
		answer = "Bye, my Lord!"
	elif intent == "greet":
		answer = "Hello, my Lord! May the force be with you. Tell me... What do you need?"
	elif intent == "web_search":
		global terms_to_search
		terms_to_search = sentence.split("about", 1)[1]
		answer = "so, you want me to google " + terms_to_search + "?"
	elif intent == "affirm":
		if last_intent == "web_search":
			webbrowser.open("http://www.google.com/search?q="+terms_to_search+"")
			answer = "check your browser so..."
	
	global last_intent
	last_intent = intent

	return answer + "\n"


def main():
	last_intent = ""
	terms_to_search = ""

	what_to_say = "...\n"
	while True:
		user_input = input(what_to_say)

		# processa a informacao que o cara inputou e responde
		what_to_say = process_input_sentence(user_input)


if __name__ == '__main__':
	main()
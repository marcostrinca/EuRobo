import argparse
from threading import Thread
from queue import Queue
import sys, os, time

import seeing

def process_input_sentence(sentence):
	answer = ""
	if sentence == "camera":
		seeing.init_seeing()

	return answer


def main():
    what_to_say = "Olá! Eu sou seu bot. Bora falar que eu preciso disso :)\n"
    while True:
        user_input = input(what_to_say)

        # processa a informacao que o cara inputou e responde
        what_to_say = process_input_sentence(user_input)


if __name__ == '__main__':
    main()
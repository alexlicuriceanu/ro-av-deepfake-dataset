#!/usr/bin/env bash

pkill -TERM -f 'multiprocessing.spawn.*--multiprocessing-fork'
pkill -TERM -f 'multiprocessing.resource_tracker'

ps -fu "$USER" | grep -E '[m]ultiprocessing'
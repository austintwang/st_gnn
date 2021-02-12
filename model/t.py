import sys
from model.dispatchers import Dispatcher

args = sys.argv[1:] 
name = args[0]
dname = args[1]
device = dname if dname == "cpu" else f"cuda:{dname}" 
if len(args) > 2 and args[2] == "c":
	clear_cache = True
else:
	clear_cache = False

Dispatcher.names[name].dispatch(device, clear_cache)
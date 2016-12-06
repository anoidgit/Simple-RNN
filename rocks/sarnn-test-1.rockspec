package = "sarnn"
version = "scm-1"

source = {
	url = "git://github.com/anoidgit/Simple-RNN",
	tag = "master"
}

description = {
	summary = "A fast, simple rnn library that based on torch and fit torch standard used for neural machine translation and the other task, This package may include LSTM, GRU, FastLSTM, RHN, LSTMP, etc.",
	detailed = [[
A library to based on torch.
	]],
	homepage = "https://github.com/anoidgit/Simple-RNN",
	license = "GPL"
}

dependencies = {
	"torch >= 7.0",
	"nn >= 1.0"
}

build = {
	type = "command",
	build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
	]],
	install_command = "cd build && $(MAKE) install"
}

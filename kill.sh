# shellcheck disable=SC2046
kill -9 $(pgrep xt_main)
kill -9 $(pgrep xt_explorer)
kill -9 $(pgrep xt_broker)
kill -9 $(pgrep plasma_store_se)
kill -9 $(pgrep multi_trainer)
kill -9 $(pgrep xt_compress)

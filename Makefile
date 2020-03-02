.PHONY: clean

clean:
	rm -rf active_meta_learning.egg-info && \
	find . -name "seed_42" -type d -exec rm -rf {} \;

# Makefile for Sphinx documentation with i18n support

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXINTL    ?= $(HOME)/.local/bin/sphinx-intl
SOURCEDIR     = source
BUILDDIR      = build
LANGUAGES     = zh_CN

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile gettext intl-update html-en html-zh html-all serve clean-all

# ---- Internationalization targets ----

gettext:
	@$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

intl-update: gettext
	@for lang in $(LANGUAGES); do \
		$(SPHINXINTL) update -p "$(BUILDDIR)/gettext" -l $$lang -d "$(SOURCEDIR)/locale"; \
	done

# ---- Multi-language build targets ----

html-en:
	SPHINX_LANGUAGE=en $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/en" $(SPHINXOPTS) $(O)

html-zh:
	SPHINX_LANGUAGE=zh_CN $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/zh_CN" $(SPHINXOPTS) $(O)

html-all: html-en html-zh
	@cp "$(SOURCEDIR)/_static/redirect.html" "$(BUILDDIR)/html/index.html" 2>/dev/null || true
	@echo "Built all languages. Serve with: make serve"

serve:
	@echo "Serving at http://127.0.0.1:8000"
	@cd "$(BUILDDIR)/html" && python3 -m http.server 8000

clean-all:
	rm -rf "$(BUILDDIR)"

# Default single-language build (backwards compatible)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

def extract_from_pdf_streaming(self, file_path: Path) -> Generator[str, None, None]:
    try:
        doc = pymupdf.open(file_path)
        total_pages = len(doc)

        logger.info(f"Processing PDF: {file_path.name} ({total_pages} pages)")

        for start_idx in range(0, total_pages, self.PAGE_BATCH_SIZE):
            end_idx = min(start_idx + self.PAGE_BATCH_SIZE, total_pages)
            batch_text = []

            for page_num in range(start_idx, end_idx):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()

                    # Filter out empty or very short pages
                    if len(page_text.strip()) >= self.MIN_PAGE_TEXT_LENGTH:
                        batch_text.append(page_text)

                except Exception as e:
                    logger.warning(f"Error while processing page {page_num}: {e}")
                    continue

            if batch_text:
                yield "\n".join(batch_text)

        doc.close()

    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        yield ""

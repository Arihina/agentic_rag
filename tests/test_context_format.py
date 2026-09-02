from __future__ import annotations

"""format_context: snapshot нового формата (kb-v2).

Инвариант нумерации проверяем отдельно — LLM ссылается на источник по
индексу [i], и порядок обязан совпадать с порядком в chunks. Всё
остальное — форматирование locate-строки (headings, pages, их пустые
варианты и сочетания).
"""

import unittest

from app.core.context_format import _format_location, format_context
from tests.core_fakes import hit


class LocationFormatTests(unittest.TestCase):
    """_format_location — где находится фрагмент в документе."""

    def test_both_headings_and_pages(self):
        source = {"headings": ["Раздел 1", "Пункт 1.2"], "pages": [5, 6, 7]}
        self.assertEqual(_format_location(source),
                         "Раздел 1 > Пункт 1.2 · с. 5-7")

    def test_single_page(self):
        source = {"headings": ["Введение"], "pages": [1]}
        self.assertEqual(_format_location(source), "Введение · с. 1")

    def test_pages_only(self):
        """Заголовков нет — только страницы, без ведущего пробела/точки."""
        source = {"headings": [], "pages": [12, 13]}
        self.assertEqual(_format_location(source), "с. 12-13")

    def test_headings_only(self):
        source = {"headings": ["Приложение А"], "pages": []}
        self.assertEqual(_format_location(source), "Приложение А")

    def test_empty_source(self):
        """Ни заголовков, ни страниц — пустая строка, чтобы вызывающий
        код мог решить, стоит ли рисовать пробел перед ней."""
        self.assertEqual(_format_location({}), "")
        self.assertEqual(_format_location({"headings": [], "pages": []}), "")

    def test_missing_keys(self):
        """kb-v2 гарантирует непустые массивы, но реальный OpenSearch иногда
        просто не возвращает пустое поле — код должен пережить."""
        self.assertEqual(_format_location({"pages": [3]}), "с. 3")
        self.assertEqual(_format_location({"headings": ["X"]}), "X")

    def test_headings_as_string_fallback(self):
        """Если поле headings придёт строкой, а не списком, не падаем."""
        source = {"headings": "Просто строка", "pages": []}
        self.assertEqual(_format_location(source), "Просто строка")

    def test_pages_out_of_order_are_sorted(self):
        """Порядок страниц в _source не гарантирован — сортируем перед
        min/max, чтобы вывод не путал пользователя."""
        source = {"headings": [], "pages": [7, 3, 5]}
        self.assertEqual(_format_location(source), "с. 3-7")

    def test_pages_with_bad_values_ignored(self):
        """Одно кривое значение (None, строка) — гасим весь блок страниц,
        а не поднимаем исключение в промпт-билдере."""
        source = {"headings": ["X"], "pages": [1, None, 3]}
        self.assertEqual(_format_location(source), "X")


class FormatContextTests(unittest.TestCase):
    """format_context — сборка всего контекста для LLM."""

    def test_single_chunk_with_full_metadata(self):
        chunks = [hit("c1", "Порядок действий: сначала A, потом B.",
                      headings=["Регламент", "1. Общие положения"],
                      pages=[3, 4])]
        expected = (
            "[1] Регламент > 1. Общие положения · с. 3-4\n"
            "Порядок действий: сначала A, потом B."
        )
        self.assertEqual(format_context(chunks), expected)

    def test_numbering_starts_at_one(self):
        """LLM ссылается [1], [2]... — нумерация ОБЯЗАНА начинаться с 1
        и идти в порядке chunks, иначе grounding для message_sources
        поедет."""
        chunks = [
            hit("a", "первый"),
            hit("b", "второй"),
            hit("c", "третий"),
        ]
        result = format_context(chunks)
        self.assertTrue(result.startswith("[1]"))
        self.assertIn("\n\n[2]", result)
        self.assertIn("\n\n[3]", result)

    def test_chunks_separated_by_blank_line(self):
        """Разделитель `\\n\\n` — устойчивый маркер границы фрагментов и
        для LLM, и для человека при отладке."""
        chunks = [hit("a", "первый"), hit("b", "второй")]
        result = format_context(chunks)
        self.assertIn("\n\n[2]", result,
                      "между chunks должен быть один пустой пропуск строки")
        # Ровно один пустой пропуск, не два подряд.
        self.assertNotIn("\n\n\n", result)

    def test_no_metadata_still_renders_header(self):
        """Пустые headings и pages не должны сломать формат — просто нет
        location после [i]."""
        chunks = [hit("a", "просто текст")]
        result = format_context(chunks)
        self.assertEqual(result, "[1]\nпросто текст")

    def test_empty_chunks_returns_empty_string(self):
        """Пустой ввод — пустой выход. Обработку 'нет данных' делает
        уровень выше (generate_answer), не форматер."""
        self.assertEqual(format_context([]), "")

    def test_never_leaks_content_vector(self):
        """Если по какой-то причине в hit протёк content_vector — не рисуем
        его в промпт. Форматер смотрит только на content/headings/pages —
        значит защита автоматическая, но проверим явно."""
        h = hit("a", "текст", headings=["H"], pages=[1])
        h["_source"]["content_vector"] = [0.1] * 1024
        result = format_context([h])
        self.assertNotIn("0.1", result)
        self.assertNotIn("content_vector", result)


if __name__ == "__main__":
    unittest.main()

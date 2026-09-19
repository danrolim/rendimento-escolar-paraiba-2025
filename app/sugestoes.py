"""Envio das sugestões dos usuários do painel para um Google Forms (sem senha ou credencial)."""

import urllib.parse
import urllib.request

MAX_PALAVRAS = 500
MAX_TITULO = 80
MAX_CARACTERES = 5000


def contar_palavras(texto: str) -> int:
    return len(texto.split())


def montar_dados(titulo: str, sugestao: str, campo_titulo: str, campo_sugestao: str) -> dict[str, str]:
    return {
        campo_titulo: " ".join(titulo.split())[:MAX_TITULO],
        campo_sugestao: sugestao.strip(),
    }


def enviar(url: str, dados: dict[str, str]) -> None:
    corpo = urllib.parse.urlencode(dados).encode("utf-8")
    req = urllib.request.Request(url, data=corpo, method="POST")
    with urllib.request.urlopen(req, timeout=15):
        pass

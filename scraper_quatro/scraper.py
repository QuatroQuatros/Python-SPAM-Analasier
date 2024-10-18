import asyncio

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = 'https://catalogodefraudes.rnp.br'
OUTPUT_FILE_NAME = 'test.csv'

urls = [
    f'{BASE_URL}/frauds/{id}'
    for id in range(1, 16347)
]


warning_message = '''###############################################################################
# !!! ATENCAO !!! ATENCAO !!! ATENCAO !!! ATENCAO !!! ATENCAO !!! ATENCAO !!! #
#                                                                             #
# O TEXTO ABAIXO FOI TRANSCRITO A PARTIR DE UMA FRAUDE CADASTRADA EM NOSSOS   #
# SISTEMAS ATRAVES DA COLETA DE DADOS NA INTERNET E/OU CONTRIBUICAO DE        #
# PARCEIROS E/OU USUARIOS.                                                    #
#                                                                             #
# EM CASO DE DUVIDAS ENTRE EM CONTATO ATRAVES DO EMAIL: cais@cais.rnp.br      #
#                                                                             #
# OBRIGADO.                                                                   #
#                                                                             #
# CENTRO DE ATENDIMENTO A INCIDENTES DE SEGURANCA (CAIS)                      #
# REDE NACIONAL DE ENSINO E PESQUISA (RNP)                                    #
###############################################################################'''


async def req(session: aiohttp.ClientSession, url: str):
    async with session.get(url) as response:

        if response.status != 200:
            print(f'Error: {response.status} on {url}')
            return None, url

        return await response.text(), url


def parse(html: str, url: str):

    soup = BeautifulSoup(html, 'html.parser')

    text_page = soup.find('div', attrs={'id': 'text-page'})

    if not text_page:
        return None

    h4s = text_page.find_all('h4')  # type: ignore

    title = soup.find('h2').get_text()  # type: ignore
    subject = h4s[0].get_text().split('Assunto da mensagem: ')[-1]
    date = h4s[1].get_text().split('Data de inclusão: ')[-1]
    description = text_page.find('p').get_text()  # type: ignore

    raw_content = text_page.find('pre')  # type: ignore

    if raw_content:
        content = raw_content.get_text().split(warning_message)[-1].strip()  # noqa # type: ignore
    else:
        content = None

    url_images = text_page.find_all('img', attrs={'class': 'img-responsive'})  # type: ignore

    data = {
        'url': url,
        'title': title,
        'description': description,
        'subject': subject,
        'content': content,
        'url_images': ', '.join(f'{BASE_URL}{url_image["src"]}' for url_image in url_images),
        'added_at': date,
    }

    return data


async def main():

    all_data = []

    async with aiohttp.ClientSession() as session:
        print(f'Starting {len(urls)} requests')

        tasks = [asyncio.ensure_future(req(session, url)) for url in urls]
        responses = await asyncio.gather(*tasks)

        for res, url in responses:

            if not res:
                continue

            data = parse(res, url)

            if data:
                all_data.append(data)

        print('Finished')

    print(f'Saving to {OUTPUT_FILE_NAME}')
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE_NAME, index=False)


asyncio.run(main())

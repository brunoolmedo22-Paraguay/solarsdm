# Exportação compacta para o otimizador

Foi criada a aba **5 · Exportar resultados**, separada dos gráficos de análise.

## Fluxo da aba

1. Selecionar o dia da simulação.
2. Definir a duração do intervalo em minutos.
3. Escolher o horário inicial.
4. Conferir os KPIs calculados somente para essa janela.
5. Conferir a tabela completa da janela.
6. Selecionar as colunas que entrarão no CSV.
7. Definir o nome do arquivo e baixar.

## Configuração padrão

- Duração: **120 minutos**.
- Colunas: **timestamp** e **potência gerada**.
- Nome do arquivo: **Modelo_solar_1.csv**.

Para dados de 1 minuto, uma janela de 120 minutos usa o intervalo semiaberto
`[início, fim)` e gera exatamente 120 linhas. Exemplo: 06:00 até 08:00 contém
06:00, 06:01, ..., 07:59.

## Nomes padrão do CSV

A interface apresenta nomes amigáveis, mas o arquivo usa cabeçalhos estáveis e
adequados à integração, como:

- `timestamp`
- `potencia_gerada_W`
- `energia_passo_Wh`
- `ghi_W_m2`
- `temperatura_ambiente_C`
- `temperatura_celula_C`
- `eficiencia_pct`

## Arquivos alterados

- `app.py`
- `simulation/export.py` (novo)
- `tests/test_export_results.py` (novo)

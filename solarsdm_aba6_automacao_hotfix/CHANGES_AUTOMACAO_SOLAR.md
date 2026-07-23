# Aba 6 — Automação solar

## Implementação

- Nova aba `6 · Automação`, sem campos de configuração.
- Um único botão executa os quatro CSVs de previsão armazenados em
  `Dados_exemplo/`.
- Módulo fixo: Canadian Solar `CS7L-580MS`.
- Parâmetros SDM fixos:
  - IL = 18,2996 A
  - I0 = 1,0494e-11 A
  - Rs = 0,1074 Ω
  - Rsh = 66,2 Ω
  - n = 0,9332
  - Ns = 60
- Arranjo fixo: 3 módulos em série e 2 strings em paralelo (6 módulos,
  3,480 kWp).
- Cada CSV é executado somente nos 120 timestamps contidos nele.
- Saída por caso: `Modelo_solar_HH.csv` com somente `timestamp` e
  `potencia_gerada_W`.
- Download individual e pacote ZIP com todos os resultados.
- Download de relatório Markdown com a configuração completa.

## Arquivos adicionados

- `simulation/automation.py`
- `tests/test_automation.py`
- `Dados_exemplo/PREVISAO_SOLAR_120min_06.csv`
- `Dados_exemplo/PREVISAO_SOLAR_120min_12.csv`
- `Dados_exemplo/PREVISAO_SOLAR_120min_16.csv`
- `Dados_exemplo/PREVISAO_SOLAR_120min_18.csv`

## Arquivo alterado

- `app.py`

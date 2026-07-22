# Atualização do fluxo solar — carregamento do CSV do Vitor

## O que foi implementado

1. **Detecção automática das colunas do CSV**
   - Reconhece `Timestamp`, `G`, `GHI`, `GHI_REAL`, `GHI_PREDITO` e variações.
   - Quando há série real e prevista, `GHI_PREDITO` é selecionada por padrão.
   - A interface permite trocar manualmente a coluna utilizada.

2. **Reconstrução do eixo temporal completo**
   - Infere o passo temporal do arquivo.
   - Pode completar cada dia entre `00:00` e `23:59`.
   - Valores de irradiância ausentes recebem `G = 0`.

3. **Rastreabilidade dos dados preenchidos**
   - `is_original`: linha existente no CSV.
   - `is_filled`: linha criada pela plataforma.
   - `fill_type`:
     - `original`;
     - `preenchido_noite`;
     - `preenchido_lacuna_diurna`.
   - `Tamb_filled`: informa se a temperatura foi gerada/interpolada.

4. **Mapa de cobertura temporal**
   - Verde: dados existentes no CSV.
   - Cinza: noite completada com zero.
   - Vermelho: lacuna ausente dentro da janela solar estimada.

5. **Tratamento de CSV sem temperatura**
   - Opção A: definir temperatura constante de dia e de noite.
   - Opção B: usar curva padrão sazonal, com temperatura mínima e máxima editáveis.

6. **Filtro dos gráficos por data**
   - Visualização de um dia específico.
   - Visualização de um intervalo de dias.
   - Disponível na prévia do perfil e nos resultados da simulação.

7. **Resultados para múltiplos dias**
   - KPIs passaram a considerar a duração real do período.
   - O fator de capacidade não fica mais limitado à hipótese fixa de 24 horas.
   - O gráfico de energia funciona com séries de vários dias.

8. **Exportação com rastreabilidade**
   - As colunas de origem/preenchimento acompanham a série simulada exportada.

## Verificação realizada com o arquivo recebido

Arquivo: `w5h5_1min_h1_2563.csv`

- Período real encontrado: **01/09/2016 06:45 até 30/12/2016 16:44**.
- Resolução detectada: **1 minuto**.
- Registros originais: **75.088**.
- Registros após completar os dias: **174.240**.
- Noite preenchida: **97.038 registros**.
- Lacunas dentro da janela solar estimada: **2.114 registros**.

As lacunas diurnas também são preenchidas com zero para manter o eixo regular, mas ficam destacadas em vermelho e não são tratadas silenciosamente como noite.

## Arquivos alterados

- `app.py`
- `models/irradiance_model.py`
- `simulation/mpp.py`
- `simulation/energy.py`
- `visualization/plots.py`
- `README.md`

## Arquivo novo

- `tests/test_profile_loader.py`

## Testes executados

```bash
python -m tests.validate
python -m unittest tests.test_profile_loader -v
```

Todos os testes passaram.

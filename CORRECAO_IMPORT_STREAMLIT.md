# Correção do ImportError no Streamlit

O erro `cannot import name detect_profile_columns` ocorria quando o Streamlit
mantinha em memória uma versão antiga de `models/irradiance_model.py` durante
o hot reload, ou quando `app.py` e o módulo eram enviados em commits separados.

A correção aplicada:

1. substitui o `from ... import ...` por carregamento validado do módulo;
2. invalida o cache de importação e recarrega o módulo quando a versão não coincide;
3. adiciona uma versão explícita da API do carregador de perfis;
4. produz uma mensagem clara caso os arquivos estejam realmente incompatíveis.

Suba **todo o conteúdo deste ZIP em um único commit**, mantendo `app.py` na raiz
e `irradiance_model.py` dentro da pasta `models/`. Depois faça um reboot da app.

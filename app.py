import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('transacoes.csv', encoding='latin1')

df['valor'] = df.groupby('estado_cliente')['valor'].transform(
    lambda x: x.fillna(x.median())
)  

df['plataforma'] = 'Mobile'

df['data_transacao'] = pd.to_datetime(df['data_transacao'])
df['data_transacao'] = df['data_transacao'].dt.tz_localize('America/Sao_Paulo')
df['dia_semana'] = df['data_transacao'].dt.day_name()
df['mes'] = df['data_transacao'].dt.month

df = df.drop_duplicates(keep='first')

cond_mes = df['mes'] == 9
cond_estado = (df['estado_cliente'] == 'SP') | (df['estado_cliente'] == 'RJ')
cond_valor = df['valor'] > 5000
filtro = cond_mes & cond_estado & cond_valor
df_setembro = df[filtro]

risco_dict = {
    'C100': 'Baixo',
    'C101': 'Alto',
    'C102': 'Alto em dobro',
    'C103': 'Alto ao quadrado',
    'C104': 'pneumoultramicroscopicossilicovulcanoconiotico'
}

df['nivel_risco'] = df['id_cliente'].map(risco_dict)

tabela = pd.pivot_table(
    df,
    values='valor',
    index='mes',
    columns='nivel_risco',
    aggfunc='sum',
    margins=True
)

def calcula_zscore_por_grupo(df, coluna_valor, coluna_grupo):
    media = df.groupby(coluna_grupo)[coluna_valor].transform('mean')
    desvio = df.groupby(coluna_grupo)[coluna_valor].transform('std')
    z = (df[coluna_valor] - media) / desvio
    return z

df['zscore'] = calcula_zscore_por_grupo(df, 'valor', 'estado_cliente')

df_anomalias = df[df['zscore'] > 2.5]

df['data_dia'] = df['data_transacao'].dt.date
serie_diaria = df.groupby('data_dia')['valor'].sum()

media_movel = serie_diaria.rolling(7).mean()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(serie_diaria.index, serie_diaria.values, label='Valor diario', marker='o')
ax.plot(media_movel.index, media_movel.values, label='Media movel (7 dias)', linewidth=2.5)

ax.set_ylim(bottom=0)
ax.set_title('Transacoes diarias e media movel de 7 dias')
ax.set_xlabel('Data')
ax.set_ylabel('Valor total (R$)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



'''
Prints utilizados ao decorrer da atvd
print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isna().sum())

print(df['valor'].isna().sum())

print(df.head())
print(df.dtypes)

print(df.duplicated().sum())

print(f"{df_setembro.shape} \n{"===" * 20}")
print(f"{df_setembro[['id_cliente', 'valor', 'estado_cliente', 'mes']].head()} \n{"===" * 20}")
print(f"{df_setembro['estado_cliente'].unique()} \n{"===" * 20}")
print(f"{df_setembro['valor'].min()}")

print(df[['id_cliente', 'nivel_risco']].head())
print(df['nivel_risco'].value_counts())

print(tabela)
print(tabela.shape)

print(df[['valor', 'estado_cliente', 'zscore']].head())
print(df['zscore'].describe())

print(df_anomalias.shape)
print(df_anomalias[['id_cliente', 'valor', 'estado_cliente', 'zscore']].head())
print(df_anomalias['valor'].min())
'''


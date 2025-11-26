import pandas as pd
df = pd.read_csv("C:\\Users\\palak priya\\OneDrive\\Desktop\\fraud_upi\\data\\Variant III.csv")
print("Total frauds:", df['fraud_bool'].sum())
# Top merchants/payment types with fraud percent
grp = df.groupby('payment_type').agg(total=('fraud_bool','size'), frauds=('fraud_bool','sum'))
grp['fraud_rate'] = grp['frauds']/grp['total']
print(grp.sort_values('fraud_rate', ascending=False).head())


# Use grp for summary table
import plotly.express as px


fig = px.bar(
    grp.reset_index(),
    x='payment_type',
    y='fraud_rate',
    text='fraud_rate',
    color='fraud_rate',
    color_continuous_scale='Reds',
    title='Fraud Rate by Payment Type (%)'
)
fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
fig.update_layout(title_x=0.5)
fig.show()

import plotly.figure_factory as ff


labels = ['AA', 'AB', 'AC', 'AD', 'AE']
heat_data = []
for label in labels:
    if label in grp.index:
        frauds = grp.loc[label, 'frauds']
        genuine = grp.loc[label, 'total'] - frauds
        heat_data.append([frauds, genuine])
    else:
        heat_data.append([0, 0])

fig = ff.create_annotated_heatmap(
    z=heat_data,
    x=['Fraud', 'Genuine'],
    y=['AA', 'AB', 'AC', 'AD', 'AE'],
    colorscale='Reds',
    showscale=True
)
fig.update_layout(title='Heatmap of Fraud vs Genuine Transactions by Payment Type', title_x=0.5)
fig.show()

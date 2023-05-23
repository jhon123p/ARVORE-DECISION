from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn import tree
import graphviz

# Criar o conjunto de dados
data = pd.DataFrame({
    'Outlook': ['Ensolarado', 'Ensolarado', 'Nublado', 'Chuvoso', 'Chuvoso', 'Chuvoso', 'Nublado',
                'Ensolarado', 'Ensolarado', 'Chuvoso', 'Ensolarado', 'Nublado', 'Nublado', 'Chuvoso'],
    'Temperatura': ['Quente', 'Quente', 'Quente', 'Ameno', 'Frio', 'Frio', 'Frio',
                    'Ameno', 'Frio', 'Ameno', 'Ameno', 'Ameno', 'Quente', 'Ameno'],
    'Umidade': ['Alta', 'Alta', 'Alta', 'Alta', 'Normal', 'Normal', 'Normal',
                'Alta', 'Normal', 'Normal', 'Normal', 'Alta', 'Normal', 'Alta'],
    'Vento': ['Fraco', 'Forte', 'Fraco', 'Fraco', 'Fraco', 'Forte', 'Forte',
              'Fraco', 'Fraco', 'Fraco', 'Forte', 'Forte', 'Fraco', 'Forte'],
    'Jogar Tênis': ['Não', 'Não', 'Sim', 'Sim', 'Sim', 'Não', 'Sim',
                    'Não', 'Sim', 'Sim', 'Sim', 'Sim', 'Sim', 'Não']
})

# Separar as features (atributos) e o target (atributo classe)
features = data.drop('Jogar Tênis', axis=1)
target = data['Jogar Tênis']

# Codificar os atributos categóricos usando one-hot encoding
encoder = OneHotEncoder()
features_encoded = encoder.fit_transform(features).toarray()

# Criar o classificador da árvore de decisão
clf = DecisionTreeClassifier(criterion='entropy')

# Treinar o classificador com os dados
clf.fit(features_encoded, target)

# Gerar a visualização da árvore de decisão
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=encoder.get_feature_names_out(features.columns),
                                class_names=clf.classes_, filled=True, rounded=True)

graph = graphviz.Source(dot_data)
graph.render("arvore_de_decisao")  # Salva a visualização em um arquivo "arvore_de_decisao.pdf"
graph.view()  # Abre a visualização em uma janela separada

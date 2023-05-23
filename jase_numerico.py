from sklearn.tree import DecisionTreeClassifier
import pandas as pd

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

# Criar o classificador da árvore de decisão
clf = DecisionTreeClassifier(criterion='entropy')

# Treinar o classificador com os dados
clf.fit(features, target)

# Imprimir a árvore de decisão
print(clf.tree_)

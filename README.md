# Métricas de Avaliação em Machine Learning
Métricas de avaliação são medidas quantitativas usadas para **avaliar a performance** de um modelo. Estas medidas são cruciais não só para avaliar o desempenho ao longo do tempo como também para comparar diferentes modelos, otimizar hiperparâmetros e alinhar a avaliação do seu algotirmo com os objetivos do seu problema específico. As principais métricas de avaliação são **Acurácia,  Precisão, Sensibilidade (Recall), Especificidade, F1 (F-Score) e AUC-ROC curve**. Mais adiante, será explorado o que cada uma dessas medidas significa e como podemos obtê-las. 
## Objetivos 🎯
- **Treinar uma rede neural** para classificar imagens em 2 classes: gatos e cachorros
- Avaliar a performance dessa rede neural calculando diferentes **métricas de avaliação**
- Analisar o **significado** desses resultados
## Estrutura do projeto 🗂️
- **Métricas.ipynb:** Jupyter notebook com o código utilizado e comentários
- **PetImages:** Dataset com 23.424  imagens (11.749 gatos, 11.675 cachorros)
- **Imagens**: Pasta contendo matrizes de confusão e gráficos de performance
## Recursos Utilizados ⚒️
- **Linguagem:**  Python 3.9
- **Frameworks:** TensorFlow 2.8, Keras 2.8
- **Bibliotecas:** Numpy, Matplotlib, seaborn, scikit-learn
- **Dataset:** Kaggle Cats and Dogs Dataset da Microsoft, que é uma versão mais compacta do dataset original. Disponível nesse [link](https://www.microsoft.com/en-us/download/details.aspx?id=54765). 
## Implementação 🧑‍💻
O nosso problema é uma tarefa de **classificação binária em 2 classes**, em que "cachorro" (1) é a nossa classe postitiva e "gato" (0) a classe negativa. 
Para que possamos medir métricas de avaliação, antes precisamos **contruir e treinar um modelo**. Entretando, eu irei me concentrar mais na etapa de **avaliação dos resultados**, que é o foco do projeto. 

Depois de importar as bibliotecas necessárias, defini o tamanho do dataset e o dividi em **treino (70%), validação (20%) e teste (10%)**.

A seguir eu treinei uma **rede neural** usando data augmentation e uma arquitetura robusta (8 camadas convolucionais + 2 camadas densas) para extrair features relevantes e classificar as imagens do dataset corretamente. Verifique o arquivo Métricas.ipynb para mais detalhes. Um resumo do modelo pode ser visto a seguir:
<p align="center" width="100%">
</p>
Após treinar por 30 epochs com Learning rate adaptativa obtive resultados satisfatórios na validação e salvei os melhores pesos. Então, treinei o modelo no dataset de teste, onde o modelo exibiu esses indicadores:
<p align="center" width="100%">
</p>

## Avaliação de resultados 📈
O primeiro passo para gerar as métricas de avaliação é calcular a **Matriz de Confusão**. Se trata de uma tabela que compara os valores previstos (nas colunas) com os valores reais (nas linhas) de um conjunto de dados. Desse modo, teremos quatro tipos de valores:
 - **VP** (Verdadeiros Positivos): número de previsões positivas que realmente eram positivas
 - **FP** (Falsos Positivos): número de previsões positivas que na verdade eram negativas
 - **VN** (Verdadeiros Negativos): número de previsões negativas que realmente eram negativas
 - **FN** (Falsos Negativos): número de previsões negativas que na verdade eram positivas
Para construir a matriz, temos que converter o nosso dataset de teste (um tensor) em arrays de imagens e rótulos:
<p align="center" width="100%">
</p>
Então, iremos definir as previsões do nosso modelo e classificá-las usando um **threshold padrão**, de >=0.5. Isso significa que se o nosso modelo tiver **50%** ou mais de certeza que um animal é um cachorro, esse animal será classificado como cachorro (1). Senão, o animal será classificado como gato (0). 
Para ter uma visão completa da **distribuição das classificações** em VP, VN, FP e FN eu criei duas matrizes de confusão, uma com proporções e outra com valores absolutos:
<p align="center" width="100%">
</p>
Usando esses valores absolutos podemos calcular manualmente cada métrica usando suas respectivas fórmulas:
- **Acurácia** =   (vp + vn) / (vp + vn + fp + fn) <br>
	A proporção de previsões corretas do total de elementos. 	
- **Precisão** = vp / (vp + fp) if (vp + fp) > 0 else 0.0  
	A proporção de previsões positivas verdadeiras do total de previsões positivas. 
- **Sensibilidade (Recall)** = vp / (vp + fn) if (vp + fn) > 0 else 0.0  
	A proporção de previsões positivas verdadeiras to total de instâncias positivas. 
-  **Especificidade** = vn / (vn + fp) if (vn + fp) > 0 else 0.0
	A proporção de previsões negativas verdadeiras do total de instâncias negativas. 
-  **F1 (F-Score)** = (2*vp) / (2*vp + fp + fn) if (2*vp + fp + fn) > 0 else 0.0 
	A média harmônica entre Precisão e Recall, útil para datasets desbalanceados. 
*Obs: if > 0 else 0.0 evita divisão por zero.

Todas as métricas calculadas aqui dependem do threshold específico que definimos (>=0.5).
**Resultados:**
- **Acurácia:** Em 92% das imagens o modelo acerta a classe.
- **Precisão:** Entre todas as imagens que o modelo previu como "cão", 91% realmente são cães. Se o modelo rotula "cão", há alta confiança de que é cão. 
- **Sensibilidade (Recall):** De todos os cães do dataset, o modelo identifica 92% como "cão". Poucos cães são classificados errôneamente como gatos. 
- **Especificidade:** De todos os gatos do dataset, o modelo identifica 92% como "gato". Poucos gatos são classificados errôneamente como cães. 
- **F1:** Bom equilíbrio entre não acusar gato de ser cão (FP) e não deixar cães passarem despercebidos (FN). 

Por fim, para medir a habilidade do classificador de **distinguir entre as classes** vamos visualizar a curva AUC-ROC. Essa métrica não depende de um threshold específico, ela gera uma visão geral da performance do modelo ao longo de todos os thresholds, sendo muito útil para quando temos classes desbalanceadas. As medidas utilizadas para calcular a curva são: TPR (True Positive rate), que é o mesmo que Recall e FPR, que representa com qual frequência o modelo classifica incorretamente instâncias negativas como positivas. Quanto mais **próxima de 1.0** a TPR for, e **maior a área sob a curva** mais preciso o modelo.
## Conclusão 🐱🐶
Este projeto possibilitou uma compreensão mais profunda sobre a **importância** das métricas de avaliação na análise de modelos de inteligência artificial. Mais do que apenas treinar uma rede neural, o foco esteve em **interpretar seus resultados** e entender como medidas como acurácia, precisão, recall, especificidade, F1 e AUC-ROC refletem diferentes aspectos da performance. Essas métricas são fundamentais para diagnosticar erros, comparar modelos e alinhar a escolha do classificador com os **objetivos do problema real**, tornando-se indispensáveis no processo de desenvolvimento de soluções robustas em machine learning.

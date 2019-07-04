# Reconhecimento-de-Faces
Aplicação da biblioteca dlib para identificação e reconhecimento de faces

As bibliotecas OpenCV e Dlib possuem um grande poder de detecção e reconhecimento de faces humanas, dentre outras capacidades.
Nesse repositório demonstro como tal biblioteca aprende com determinada imagem e a reconhece em outra imagem diferente.

1˚: Arquivo treinamentofacesLuNina.py
Ele efetua o treinamento utilizando técnicas de machine learning, como SVM, a partir de imagens das pessoas que se deseja identificar. Aqui, inclui fotos da Luíza (minha filha) e da Maria (minha esposa).
O resultado é a geração dos arquivos descritores_luenina.npy e indices_luenina.pickle. Eles serão necessários na realização dos testes.

2˚: testefacesLuNina.py
Com poucas linhas de código, o submeti à fotos novas para verificar se ele identifica e retorna o nome delas.

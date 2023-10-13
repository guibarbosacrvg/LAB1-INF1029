1:
Realização de Pipeline - consiste em possibilitar que um processador trabalhe simultaneamente em diversas instruções ao executar um estágio diferente de cada instrução ao mesmo tempo.

Predição de desvio - o processador antecipa o código de instrução através da memória e prediz qual desvio ou quais instruções serão executadas em seguida. A predição potencialmente aumenta a quantidade de trabalhos disponíveis para o processador executar.

Execução Superescalar - Capacidade de enviar mais de uma instrução em todos os ciclos de clock do processador, o que requer mais de um pipeline, usando pipelines paralelos.

Análise de fluxo de dados - O processador analisa quais instruções são dependentes de resultados de instruções anteriores, visando criar uma lista otimizada de instruções. As instruções são executadas quando prontas, independente do pedido do programa inicial, para prevenir atrasos desnecessários.

Execução especulativa - Usando a predição de desvio e a análise de fluxo de dados em conjunto, alguns processadores realizam instruções antes de seu surgimento real na execução do programa, mantendo os resultados em um local distinto.

Hazards de pipeline - Ocorre quando o pipeline, ou alguma parte dele, deve parar porque as condições não permitem execução contínua. Essas paradas são conhecidas como bolhas de pipeline.

Hazard de recurso - Ocorre quando pelo menos duas instruções que estão no pipeline precisam do mesmo recurso, fazendo com que as instruções sejam executadas em série ao invés de serem executadas em paralelo.

Hazard de dados - Ocorre quando há um conflito no acesso do operando, quando duas instruções estão em sequência e tentam acessar a mesma região de memória ou registrador. Caso não ocorra um atraso proposital no pipeline, o programa pode produzir um resultado incorreto.

Hazard de controle - Ocorre quando o pipeline tenta prever uma instrução de desvio mas erra, fazendo com que instruções que foram inseridas no pipeline sejam descartadas em seguida. É chamado também de hazard de desvio.

Registradores - São divididos em dois grupos: Registradores visíveis ao usuário e registradores de controle e de estado.

Registradores visíveis ao usuário - possibilitam que o programador minimize os acessos à memória principal, maximizando o uso de registradores.

Registradores de controle e de estado - usados pela unidade de controle para controlar a operação do processador e por programas privilegiados do SO para controlar a execução de programas.

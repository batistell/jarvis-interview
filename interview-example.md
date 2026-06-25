# 👤 Prep de Entrevista - Face Registry (Estilo de Respostas)

Este arquivo serve como referência de estilo e conteúdo para o `jarvis-interview`. Ele contém exemplos reais de perguntas técnicas de entrevista e as respostas correspondentes baseadas no projeto **Face Registry**.

---

## ❓ Perguntas de Entrevista & Respostas Técnicas do Candidato

### Pergunta 1: Por que você escolheu rodar PyTorch na JVM usando DJL em vez de fazer uma API REST em Python (como Flask/FastAPI), que é o padrão da indústria para IA?
**Resposta:** 
> "A criação de um microserviço Python separado traria três grandes problemas para este ecossistema: latência de rede, sobrecarga de infraestrutura e complexidade de deploy. Ao trafegar imagens pesadas via REST/gRPC entre a JVM e o Python, adicionaríamos latência de serialização/deserialização e transporte. Além disso, teríamos que gerenciar mais um container em produção, dividindo recursos de CPU/RAM.
>
> Escolhi o **Deep Java Library (DJL)** da AWS com o motor PyTorch nativo via **JNI (Java Native Interface)** porque ele carrega as bibliotecas dinâmicas do **PyTorch** nativo de forma automática para o sistema operacional hospedado. A inferência ocorre in-process com desempenho idêntico ao C++ nativo e sem custos de rede. Isso reduz o custo de infraestrutura e simplifica a implantação, mantendo apenas um container para o backend."

---

### Pergunta 2: Como você tratou o risco de OOM (Out Of Memory) na inicialização da aplicação ao carregar milhares de usuários do banco para o cache em memória?
**Resposta:**
> "Para alimentar o cache biométrico em memória sem sobrecarregar a JVM, eu utilizei duas técnicas fundamentais:
> 1. **Projeção JPQL Otimizada (UserLightweight):** O banco de dados PostgreSQL armazena as fotos como `BYTEA` (que podem ter até 10MB por registro). Se usássemos o `findAll()` padrão do JPA, o Hibernate traria todos os arrays de bytes das imagens para a memória heap, causando um estouro de memória instantâneo com milhares de registros. Eu criei a projeção `UserLightweight` que seleciona apenas `id`, `cpf`, `name` e `embedding` do banco via query customizada, ignorando a coluna de imagem.
> 2. **Cache Otimizado:** O cache armazena apenas as referências de texto e o array de floats de 512 posições. O consumo de memória por usuário no cache é de aproximadamente 2.2 KB. Com isso, conseguimos manter mais de 100.000 usuários em memória gastando menos de 250 MB de heap."

---

### Pergunta 3: Qual é a diferença matemática e de performance entre a Similaridade de Cosseno tradicional e o Produto Escalar (Dot Product) que você implementou?
**Resposta:**
> "A similaridade de cosseno matemática entre dois vetores $A$ e $B$ é dada pela fórmula:
> $$\text{similaridade} = \frac{A \cdot B}{\|A\| \|B\|}$$
> Onde $\|A\|$ e $\|B\|$ são as normas Euclidiana (L2) dos vetores, que exigem calcular a soma dos quadrados de todas as 512 posições e extrair a raiz quadrada de cada vetor, seguido por uma divisão em ponto flutuante.
>
> No meu projeto, eu normalizo os embeddings gerados pelo FaceNet aplicando a normalização L2 logo após a extração, garantindo que a norma do vetor seja exatamente igual a 1 ($\|A\| = 1$ e $\|B\| = 1$). Quando os vetores são normalizados, o denominador da fórmula se torna $1 \cdot 1 = 1$. Portanto, a similaridade de cosseno passa a ser equivalente ao **Produto Escalar (Dot Product)**:
> $$\text{similaridade} = A \cdot B = \sum_{i=1}^{512} (A_i \cdot B_i)$$
> Isso elimina completamente as operações custosas de raiz quadrada (`Math.sqrt`) e divisão float no loop de busca 1:N. Fazemos apenas 512 multiplicações e somas na CPU, o que aumenta o throughput de busca biométrica drasticamente."

---

### Pergunta 4: Por que você usou um limite de 5.000 usuários para alternar entre busca linear e busca paralela (ForkJoin) na identificação 1:N?
**Resposta:**
> "A paralelização de tarefas usando o pool de threads do ForkJoin no Java (via Parallel Streams ou CompletableFuture) introduz um custo de overhead devido ao agendamento de threads, divisão do array (split) e coordenação de junção dos resultados (join). 
> Para bases de dados pequenas (abaixo de 5.000 usuários), a busca sequencial linear em uma única thread é mais rápida, pois o tempo de processamento dos cálculos é menor do que o overhead necessário para gerenciar múltiplas threads.
> Quando a base escala além desse limiar, o benefício de distribuir o processamento biométrico $O(N)$ nos núcleos da CPU supera o custo de overhead das threads. Por isso, dividimos a lista em pedaços dinâmicos baseados no número de processadores disponíveis (`Runtime.getRuntime().availableProcessors()`) e calculamos de forma paralela."

---

### Pergunta 5: Como você garantiu a consistência dos dados de biometria caso dois administradores tentem atualizar a foto do mesmo usuário simultaneamente?
**Resposta:**
> "Eu utilizei um mecanismo de **Lock Pessimista de Escrita (Pessimistic Write Lock)** na camada de banco de dados do JPA:
> ```java
> @Lock(LockModeType.PESSIMISTIC_WRITE)
> @Query("SELECT u FROM User u WHERE u.cpf = :cpf")
> Optional<User> findByCpfWithLock(@Param("cpf") String cpf);
> ```
> Quando a transação de atualização do usuário A inicia e busca o usuário com esse método, o banco de dados PostgreSQL bloqueia o registro usando uma cláusula `SELECT ... FOR UPDATE`. Se o usuário B tentar atualizar o mesmo CPF ao mesmo tempo, a sua thread ficará bloqueada aguardando a transação de A terminar (fazer commit ou rollback). Isso impede condições de corrida de escrita."

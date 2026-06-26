# 👤 Prep de Entrevista - Face Registry (Estilo de Respostas)

Este arquivo serve como referência de estilo e conteúdo para o `jarvis-interview`. Ele contém exemplos reais de perguntas técnicas de entrevista e as respostas correspondentes baseadas no projeto **Face Registry**.

---

## ❓ Perguntas de Entrevista & Respostas Técnicas do Candidato

### Pergunta 1: Por que você escolheu rodar PyTorch na JVM usando DJL em vez de fazer uma API REST em Python (como Flask/FastAPI), que é o padrão da indústria para IA?
**Resposta:** 
> - **In-process vs REST**: Evita latência de rede e de serialização pesada (usa JNI nativo do PyTorch/C++).
> - **Infraestrutura**: Apenas um único container Java rodando tudo (sem duplicar custos de RAM/CPU com Flask/FastAPI).
> - **DJL (AWS)**: Carrega dependências nativas automaticamente via JNI, garantindo desempenho igual ao C++ nativo.

---

### Pergunta 2: Como você tratou o risco de OOM (Out Of Memory) na inicialização da aplicação ao carregar milhares de usuários do banco para o cache em memória?
**Resposta:**
> - **Projeção JPQL Otimizada**: Criei a projeção `UserLightweight` que busca apenas `id`, `cpf`, `name` e `embedding`. Ignoro a coluna de imagem (`BYTEA` de até 10MB), evitando sobrecarregar a Heap.
> - **Cache Otimizado**: Mantemos em memória apenas metadados básicos e o array de floats de 512 posições (2.2 KB por usuário). Com isso, 100 mil registros consomem menos de 250 MB.

---

### Pergunta 3: Qual é a diferença matemática e de performance entre a Similaridade de Cosseno tradicional e o Produto Escalar (Dot Product) que você implementou?
**Resposta:**
> - **Normalização L2 prévia**: Aplico a normalização nos embeddings logo após a extração do FaceNet, garantindo que a norma dos vetores seja exatamente 1.
> - **Equivalência**: Se a norma é 1, o denominador da similaridade de cosseno vira 1, tornando-a idêntica ao Produto Escalar ($\sum A_i \cdot B_i$).
> - **Performance**: Elimina operações caras de raiz quadrada (`Math.sqrt`) e divisão em float no loop 1:N, executando apenas multiplicações e somas na CPU.

---

### Pergunta 4: Por que você usou um limite de 5.000 usuários para alternar entre busca linear e busca paralela (ForkJoin) na identificação 1:N?
**Resposta:**
> - **Overhead de threads**: Paralelização com ForkJoin introduz custos de divisão (split), agendamento e junção (join) de threads.
> - **Até 5.000 registros**: A busca linear sequencial é mais rápida pois o processamento total é menor do que o custo de coordenar o pool de threads.
> - **Acima de 5.000**: O benefício de calcular em paralelo nos múltiplos núcleos da CPU (`availableProcessors()`) supera o overhead de gerenciar as threads.

---

### Pergunta 5: Como você garantiu a consistência dos dados de biometria caso dois administradores tentem atualizar a foto do mesmo usuário simultaneamente?
**Resposta:**
> - **Lock Pessimista de Escrita**: Uso a anotação `@Lock(LockModeType.PESSIMISTIC_WRITE)` na busca do CPF.
> - **SELECT FOR UPDATE**: Bloqueia o registro no PostgreSQL no início da transação.
> - **Segurança**: Outras transações simultâneas no mesmo CPF ficam bloqueadas esperando a primeira fazer commit ou rollback, impedindo race conditions.

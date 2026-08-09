import numpy as np
import json
import os
import random
import time
from copy import deepcopy

from src.evaluator import evaluate_population


# ─────────────────────────────────────────────────────────────
# NSGA-II Configuration
# ─────────────────────────────────────────────────────────────

N_GENES        = 8       # number of hidden layers → chromosome length
N_ALLELES      = 3       # {0: tanh, 1: sine, 2: swish}
POP_SIZE       = 8       # population size (must be even)
N_GENERATIONS  = 6       # number of generations
CROSSOVER_PROB = 0.9     # probability of crossover per pair
MUTATION_PROB  = 0.25    # probability of mutating each gene
TOURNAMENT_K   = 2       # tournament selection size
SEED           = 42

# Diversity control
MIN_UNIQUE_FRACTION = 0.5  # if fewer than this fraction of population is unique, inject immigrants
N_IMMIGRANTS         = 2   # number of fresh random chromosomes injected when diversity collapses


# ─────────────────────────────────────────────────────────────
# Chromosome utilities
# ─────────────────────────────────────────────────────────────

def random_chromosome(rng):
    """Return a random chromosome of length N_GENES."""
    return [rng.randint(0, N_ALLELES - 1) for _ in range(N_GENES)]


def initialize_population(rng):
    """Create initial population of POP_SIZE random chromosomes."""
    return [random_chromosome(rng) for _ in range(POP_SIZE)]


# ─────────────────────────────────────────────────────────────
# Dominance and sorting
# ─────────────────────────────────────────────────────────────

def dominates(a, b):
    """
    Return True if individual a dominates individual b.
    a dominates b if a is no worse in all objectives and strictly
    better in at least one.

    Parameters
    ----------
    a, b : dict with keys 'f1' and 'f2' (both lower is better)
    """
    a_no_worse = (a['f1'] <= b['f1']) and (a['f2'] <= b['f2'])
    a_better   = (a['f1'] <  b['f1']) or  (a['f2'] <  b['f2'])
    return a_no_worse and a_better


def fast_non_dominated_sort(population):
    """
    Deb et al. (2002) fast non-dominated sort.

    Parameters
    ----------
    population : list of dicts, each with 'f1', 'f2', 'chromosome'

    Returns
    -------
    fronts : list of lists of indices
             fronts[0] = Pareto front (rank 1), fronts[1] = rank 2, ...
    """
    n = len(population)
    domination_count = [0] * n       # how many individuals dominate i
    dominated_by     = [[] for _ in range(n)]  # who i dominates
    fronts           = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif dominates(population[j], population[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i]['rank'] = 0
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j]['rank'] = current_front + 1
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # remove empty last front


def crowding_distance(population, front_indices):
    """
    Compute crowding distance for individuals in a front.
    Individuals at the boundary of the front get infinite distance.

    Parameters
    ----------
    population     : full population list
    front_indices  : indices of individuals in this front
    """
    n = len(front_indices)
    if n <= 2:
        for i in front_indices:
            population[i]['crowding'] = float('inf')
        return

    for i in front_indices:
        population[i]['crowding'] = 0.0

    for obj in ['f1', 'f2']:
        sorted_front = sorted(front_indices, key=lambda i: population[i][obj])

        population[sorted_front[0]]['crowding']  = float('inf')
        population[sorted_front[-1]]['crowding'] = float('inf')

        obj_min = population[sorted_front[0]][obj]
        obj_max = population[sorted_front[-1]][obj]
        obj_range = obj_max - obj_min if obj_max != obj_min else 1e-12

        for k in range(1, n - 1):
            prev_val = population[sorted_front[k - 1]][obj]
            next_val = population[sorted_front[k + 1]][obj]
            population[sorted_front[k]]['crowding'] += (next_val - prev_val) / obj_range


# ─────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────

def crowded_comparison(a, b):
    """
    NSGA-II crowded comparison operator.
    Prefer lower rank; break ties by higher crowding distance.
    """
    if a['rank'] < b['rank']:
        return a
    elif b['rank'] < a['rank']:
        return b
    elif a['crowding'] >= b['crowding']:
        return a
    else:
        return b


def tournament_select(population, rng, k=TOURNAMENT_K):
    """
    Binary tournament selection using crowded comparison.
    Randomly pick k candidates and return the best one.
    """
    candidates = rng.choices(population, k=k)
    winner = candidates[0]
    for c in candidates[1:]:
        winner = crowded_comparison(winner, c)
    return winner


# ─────────────────────────────────────────────────────────────
# Crossover and Mutation
# ─────────────────────────────────────────────────────────────

def uniform_crossover(parent_a, parent_b, rng, prob=CROSSOVER_PROB):
    """
    Uniform crossover: each gene is taken from parent_a or parent_b
    with equal probability.

    Parameters
    ----------
    parent_a, parent_b : chromosomes (list of ints)
    prob               : probability of performing crossover at all

    Returns
    -------
    child_a, child_b : two offspring chromosomes
    """
    if rng.random() > prob:
        return deepcopy(parent_a), deepcopy(parent_b)

    child_a, child_b = [], []
    for gene_a, gene_b in zip(parent_a, parent_b):
        if rng.random() < 0.5:
            child_a.append(gene_a)
            child_b.append(gene_b)
        else:
            child_a.append(gene_b)
            child_b.append(gene_a)
    return child_a, child_b


def mutate(chromosome, rng, prob=MUTATION_PROB):
    """
    Per-gene mutation: each gene is replaced by a random allele
    with probability prob.

    Parameters
    ----------
    chromosome : list of ints
    prob       : per-gene mutation probability

    Returns
    -------
    mutated chromosome (new list)
    """
    return [
        rng.randint(0, N_ALLELES - 1) if rng.random() < prob else gene
        for gene in chromosome
    ]


# ─────────────────────────────────────────────────────────────
# Offspring generation
# ─────────────────────────────────────────────────────────────

def make_offspring(population, rng, n_offspring=POP_SIZE):
    """
    Generate n_offspring new chromosomes from the current population
    using tournament selection + uniform crossover + mutation.
    """
    offspring_chromosomes = []
    while len(offspring_chromosomes) < n_offspring:
        parent_a = tournament_select(population, rng)['chromosome']
        parent_b = tournament_select(population, rng)['chromosome']
        child_a, child_b = uniform_crossover(parent_a, parent_b, rng)
        offspring_chromosomes.append(mutate(child_a, rng))
        if len(offspring_chromosomes) < n_offspring:
            offspring_chromosomes.append(mutate(child_b, rng))
    return offspring_chromosomes[:n_offspring]


# ─────────────────────────────────────────────────────────────
# Diversity control
# ─────────────────────────────────────────────────────────────

def deduplicate_chromosomes(chromosomes, rng):
    """
    Given a list of chromosomes, replace any duplicate occurrence
    (beyond the first) with a fresh random chromosome. This prevents
    the search from wasting evaluation budget on repeated individuals
    and keeps genetic diversity in the pool.

    Parameters
    ----------
    chromosomes : list of chromosomes (list of list[int])
    rng         : random.Random instance

    Returns
    -------
    deduped : new list, same length, with duplicates replaced
    """
    seen = set()
    deduped = []
    for chrom in chromosomes:
        key = tuple(chrom)
        if key in seen:
            new_chrom = random_chromosome(rng)
            attempts = 0
            while tuple(new_chrom) in seen and attempts < 10:
                new_chrom = random_chromosome(rng)
                attempts += 1
            deduped.append(new_chrom)
            seen.add(tuple(new_chrom))
        else:
            deduped.append(chrom)
            seen.add(key)
    return deduped


def inject_immigrants_if_needed(population, rng, min_unique_fraction=MIN_UNIQUE_FRACTION,
                                 n_immigrants=N_IMMIGRANTS):
    """
    Check diversity of the population by counting unique chromosomes.
    If the fraction of unique individuals falls below min_unique_fraction,
    drop the worst n_immigrants individuals (by crowded comparison order,
    i.e. highest rank / lowest crowding removed first) and return a list
    of fresh random chromosomes that need to be evaluated to refill
    the population.

    Parameters
    ----------
    population : list of evaluated individual dicts
    rng        : random.Random instance

    Returns
    -------
    (population_kept, immigrant_chromosomes)
        population_kept       : population with worst individuals removed
        immigrant_chromosomes : list of new random chromosomes to evaluate
                                 and add back to the population
    """
    unique_keys = {tuple(ind['chromosome']) for ind in population}
    unique_fraction = len(unique_keys) / len(population)

    if unique_fraction >= min_unique_fraction:
        return population, []

    # Best-first order: lowest rank first, then highest crowding first
    sorted_pop = sorted(
        population,
        key=lambda ind: (ind.get('rank', 0), -ind.get('crowding', 0.0))
    )
    keep_count = max(1, len(population) - n_immigrants)
    kept = sorted_pop[:keep_count]

    immigrant_chromosomes = [random_chromosome(rng) for _ in range(len(population) - keep_count)]

    print(f"  [Diversity] Only {len(unique_keys)}/{len(population)} unique chromosomes "
          f"({unique_fraction:.0%}) — injecting {len(immigrant_chromosomes)} random immigrants.")

    return kept, immigrant_chromosomes


# ─────────────────────────────────────────────────────────────
# Population survival selection
# ─────────────────────────────────────────────────────────────

def select_survivors(combined, target_size=POP_SIZE):
    """
    NSGA-II environmental selection:
    1. Sort combined population by non-dominated fronts.
    2. Fill next generation front by front.
    3. If a front doesn't fit entirely, sort it by crowding distance
       and take the best individuals.

    Parameters
    ----------
    combined     : list of evaluated individuals (parents + offspring)
    target_size  : size of next generation

    Returns
    -------
    survivors : list of target_size individuals
    """
    fronts = fast_non_dominated_sort(combined)

    for front in fronts:
        crowding_distance(combined, front)

    survivors = []
    for front in fronts:
        if len(survivors) + len(front) <= target_size:
            survivors.extend(front)
        else:
            remaining = target_size - len(survivors)
            sorted_front = sorted(
                front,
                key=lambda i: combined[i]['crowding'],
                reverse=True
            )
            survivors.extend(sorted_front[:remaining])
            break

    return [combined[i] for i in survivors]


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    """Save NSGA-II state to JSON after each generation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def load_checkpoint(path):
    """Load NSGA-II state from JSON to resume after session crash."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────
# Main NSGA-II loop
# ─────────────────────────────────────────────────────────────

def run_nsga2(X_u, u_train, X_f, X_star, u_star, lb, ub,
              device='cpu',
              checkpoint_path='results/nsga2_run/checkpoint.json',
              history_path='results/nsga2_run/history.json'):
    """
    Run NSGA-II to find the Pareto-optimal set of heterogeneous
    PINN activation-function chromosomes.

    Diversity safeguards included:
    - Duplicate chromosomes within a batch of new candidates (offspring +
      immigrants), or that already exist in the current population, are
      replaced with fresh random chromosomes before evaluation. This
      guarantees no evaluation budget is spent re-training an identical
      architecture within the same generation.
    - If population diversity collapses (fewer than MIN_UNIQUE_FRACTION
      unique chromosomes), the worst individuals are dropped and replaced
      with random immigrants to keep the search exploring.

    Parameters
    ----------
    X_u, u_train, X_f : training data tensors
    X_star, u_star     : evaluation grid tensors
    lb, ub             : domain bounds (numpy arrays)
    device             : 'cpu' or 'cuda'
    checkpoint_path    : save/resume path for current population
    history_path       : save path for full generation history

    Returns
    -------
    pareto_front : list of non-dominated individuals from final population
                   each is a dict with 'chromosome', 'f1', 'f2', 'info'
    history      : list of per-generation summaries
    """
    rng = random.Random(SEED)
    np.random.seed(SEED)

    history = []
    start_gen = 0
    population = []

    # ── Resume from checkpoint if available ──────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        population = checkpoint['population']
        start_gen  = checkpoint['generation'] + 1
        history    = checkpoint.get('history', [])
        print(f"Resumed from checkpoint: generation {start_gen - 1} "
              f"({len(population)} individuals)")
    else:
        # ── Generation 0: random initialization ──────────────
        print("Initializing random population ...")
        chromosomes = initialize_population(rng)
        chromosomes = deduplicate_chromosomes(chromosomes, rng)

        results = evaluate_population(
            chromosomes, X_u, u_train, X_f, X_star, u_star,
            lb, ub, device=device,
            checkpoint_path=checkpoint_path.replace('.json', '_gen0_eval.json')
        )
        population = [
            {
                'chromosome': r['info']['chromosome'],
                'f1':         r['f1'],
                'f2':         r['f2'],
                'info':       r['info'],
                'rank':       0,
                'crowding':   0.0,
            }
            for r in results
        ]

        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance(population, front)

        gen_summary = _generation_summary(population, generation=0)
        history.append(gen_summary)
        _print_generation(gen_summary, population=population)

        save_checkpoint(
            {'generation': 0, 'population': population, 'history': history},
            checkpoint_path
        )

    # ── Main generational loop ────────────────────────────────
    for gen in range(start_gen, N_GENERATIONS):
        gen_start = time.time()
        print(f"\n{'='*50}")
        print(f"Generation {gen + 1} / {N_GENERATIONS}")
        print(f"{'='*50}")

        # 0. Diversity check on current population before breeding.
        #    If diversity has collapsed, drop the worst individuals and
        #    queue random immigrants to be evaluated alongside offspring.
        population, immigrant_chromosomes = inject_immigrants_if_needed(population, rng)

        # 1. Generate offspring chromosomes (always breed a full new batch)
        offspring_chromosomes = make_offspring(population, rng, n_offspring=POP_SIZE)

        # Combine offspring with immigrants (immigrants add extra fresh exploration)
        candidate_chromosomes = offspring_chromosomes + immigrant_chromosomes

        # 2. Remove duplicates within this batch AND against the current
        #    population, replacing them with fresh random chromosomes so we
        #    never spend a full training run re-evaluating an architecture
        #    we already have a result for.
        existing_keys = {tuple(ind['chromosome']) for ind in population}
        deduped_candidates = []
        seen_this_batch = set(existing_keys)
        for chrom in candidate_chromosomes:
            key = tuple(chrom)
            if key in seen_this_batch:
                new_chrom = random_chromosome(rng)
                attempts = 0
                while tuple(new_chrom) in seen_this_batch and attempts < 10:
                    new_chrom = random_chromosome(rng)
                    attempts += 1
                deduped_candidates.append(new_chrom)
                seen_this_batch.add(tuple(new_chrom))
            else:
                deduped_candidates.append(chrom)
                seen_this_batch.add(key)

        # 3. Evaluate offspring/immigrants (the evaluator itself also caches
        #    identical chromosomes within a single call as an extra guard)
        offspring_results = evaluate_population(
            deduped_candidates, X_u, u_train, X_f, X_star, u_star,
            lb, ub, device=device,
            checkpoint_path=checkpoint_path.replace(
                '.json', f'_gen{gen+1}_eval.json'
            )
        )

        offspring = [
            {
                'chromosome': r['info']['chromosome'],
                'f1':         r['f1'],
                'f2':         r['f2'],
                'info':       r['info'],
                'rank':       0,
                'crowding':   0.0,
            }
            for r in offspring_results
        ]

        # 4. Combine parents + offspring
        combined = population + offspring

        # 5. Select survivors for next generation
        population = select_survivors(combined, target_size=POP_SIZE)

        # 6. Log and checkpoint
        gen_summary = _generation_summary(population, generation=gen + 1,
                                          elapsed=time.time() - gen_start)
        history.append(gen_summary)
        _print_generation(gen_summary, population=population)

        save_checkpoint(
            {'generation': gen + 1, 'population': population, 'history': history},
            checkpoint_path
        )

    # ── Extract final Pareto front ────────────────────────────
    fronts = fast_non_dominated_sort(population)
    pareto_front = [population[i] for i in fronts[0]]
    pareto_front.sort(key=lambda x: x['f1'])

    # Save final results
    final_results = {
        'pareto_front': pareto_front,
        'history':      history,
        'config': {
            'pop_size':       POP_SIZE,
            'n_generations':  N_GENERATIONS,
            'crossover_prob': CROSSOVER_PROB,
            'mutation_prob':  MUTATION_PROB,
            'seed':           SEED,
        }
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"NSGA-II complete.")
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"{'='*50}")
    for ind in pareto_front:
        print(f"  {ind['chromosome']}  f1={ind['f1']:.4e}  f2={ind['f2']:.4e}")

    return pareto_front, history


# ─────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────

def _generation_summary(population, generation, elapsed=None):
    """Build a concise summary dict for one generation."""
    f1_values = [ind['f1'] for ind in population]
    f2_values = [ind['f2'] for ind in population]
    ranks     = [ind.get('rank', 0) for ind in population]
    n_front1  = sum(1 for r in ranks if r == 0)
    unique_chromosomes = len({tuple(ind['chromosome']) for ind in population})

    summary = {
        'generation':  generation,
        'f1_min':      min(f1_values),
        'f1_mean':     float(np.mean(f1_values)),
        'f2_min':      min(f2_values),
        'f2_mean':     float(np.mean(f2_values)),
        'pareto_size': n_front1,
        'unique_chromosomes': unique_chromosomes,
    }
    if elapsed is not None:
        summary['elapsed_sec'] = round(elapsed, 1)
    return summary


def _print_generation(summary, population=None):
    g   = summary['generation']
    f1m = summary['f1_min']
    f2m = summary['f2_min']
    ps  = summary['pareto_size']
    uc  = summary.get('unique_chromosomes', '—')
    t   = summary.get('elapsed_sec', '—')
    print(f"\n  Gen {g:2d} | best f1={f1m:.4e} | best f2={f2m:.4e} | "
          f"pareto_size={ps} | unique={uc} | time={t}s")

    if population is not None:
        sorted_pop = sorted(population, key=lambda x: x['f1'])[:5]
        print(f"  Top 5 chromosomes:")
        for ind in sorted_pop:
            print(f"    {ind['chromosome']}  f1={ind['f1']:.4e}  f2={ind['f2']:.4e}")
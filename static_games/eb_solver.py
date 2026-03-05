from .ProspectTheory import ProspectTheory
import functools
import jax
import jax.numpy as np
import itertools
import matplotlib.pyplot as plt

def compute_eb_equilibrium(U, pt, p1_type, p2_type):
    """Compute the cpt equilibrium using the semi smooth newton method.
       First, check for pure equilibria, then check for mixed strategies with newton""" 
    # Define equilibria variables
    equil = dict()

    row_a_1, row_a_2 = np.array([U[0, 0, 0], U[0, 1, 0]]), np.array([U[1, 0, 0], U[1, 1, 0]])
    col_a_1, col_a_2 = np.array([U[0, 0, 1], U[1, 0, 1]]), np.array([U[0, 1, 1], U[1, 1, 1]])

    x_r_a_1 = np.argsort(row_a_1)

    x_r_a_2 = np.argsort(row_a_2)

    x_c_a_1 = np.argsort(col_a_1)

    x_c_a_2 = np.argsort(col_a_2)

    sort_orders = (x_r_a_1, x_r_a_2, x_c_a_1, x_c_a_2)

    is_pt_p1, is_pt_p2 = True, False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_vals = np.linspace(0, 1, 201)

    # F over p with q fixed
    y_vals_p = np.array([compute_F(np.array([xi, 0.5]), U, is_pt_p1, is_pt_p2, pt, sort_orders) for xi in x_vals])
    ax1.plot(x_vals, y_vals_p[:, 0], label='Player 1')
    ax1.plot(x_vals, y_vals_p[:, 1], label='Player 2')
    ax1.axhline(0, color='k', linestyle='--')
    ax1.set_xlabel('p')
    ax1.set_ylabel('F_z')
    ax1.set_title('F_z over p (q=0.5)')
    ax1.legend()

    # F over q with p fixed
    y_vals_q = np.array([compute_F(np.array([0.5, xi]), U, is_pt_p1, is_pt_p2, pt, sort_orders) for xi in x_vals])
    ax2.plot(x_vals, y_vals_q[:, 0], label='Player 1')
    ax2.plot(x_vals, y_vals_q[:, 1], label='Player 2')
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlabel('q')
    ax2.set_ylabel('F_z')
    ax2.set_title('F_z over q (p=0.5)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


    # Start with many seeds to explore the space (why not?)
    # The point is to make sure we aren't convrging to specific basins based on initialization
    starting_points = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    init_list = itertools.product(starting_points, starting_points) 

    seed = 42

    key = jax.random.PRNGKey(seed)
 

    # Iterate through seed pairs, append to mixed equil
    for p, q in init_list:
        # Stopping condition
        max_tries = 1000
        counter = 0
        z = np.array([p, q])
        print(f'\r\033[K P: {p}, Q: {q}', end="")
        while counter < max_tries:
   
            counter += 1
            key, subkey = jax.random.split(key)

            try:
                z, is_root = semismooth_newton(U, z, counter, subkey, p1_type, p2_type, pt, sort_orders)
                is_root = bool(is_root)

            except ValueError as e:
                print(f'Semismooth failed, error: {e}')
                break

            if is_root:
                z_round = np.round(z, 3)
                dict_key = f'equil = {tuple(z_round)}'
                if dict_key not in equil.keys():
                    equil[dict_key] = [f'p={p}, q={q}']
                else:
                    equil[dict_key].append(f'p={p}, q={q}')
                break

    print()
    return equil


def semismooth_newton(U, z, step_num, key, p1_type, p2_type, pt, sort_orders, eps=1e-3):
    '''
    A semismooth newton solver for cpt equilibrium of beliefs. We are looking for zeros in the F(z) function
    Here F(z) is the indifference (optimality?) condition val(action 1) = val (action 2), and we are looking for 
    the point where both player 1 and 2 are indifferent
    
    The basic formula is:
    1) initialize z = p, q
    2) get initial F(z) (if F(z) = 0, return
    3) Comute Jacobian (derivative matrix to see how small perturbations in p, q influence F)
        a) used finite difference, not anything analytical
    4) solve linear equation x = J^-1 (-F(z)) to find where the function hits 0
    5) update z with x, repeat if needed (outer loop)
    '''
    # Step 1) Map the probabilities z = p, q into F space
    # F is the difference vector between actions for each player
    # That is, CPT( Value_A | prob) - CPT (Value_B | prob)
    is_pt_p1 = True if p1_type == "PT" else False
    is_pt_p2 = True if p2_type == "PT" else False

    return newton_step(z, U, step_num, key, is_pt_p1, is_pt_p2, pt, sort_orders) 

@functools.partial(jax.jit, static_argnums=(4,5,6))
def newton_step(z, U, step_num, key, is_pt_p1, is_pt_p2, pt, sort_orders, eps=1e-3):
    z = np.clip(z, 0, 1)
    F_z = compute_F(z, U, is_pt_p1, is_pt_p2, pt, sort_orders)

    # Step 2) Compute the Jacobian J for the mapping p,q -> F
    jacobian = jax.jacobian(compute_F)(z, U, is_pt_p1, is_pt_p2, pt, sort_orders)

    # Step 3) Solve for delta s.t. Jdelta = -F(z)

    # We were having a problem with the EU case where F_Z | EU is a constant value
    # with no root, so solutions were diverging. 

    # The logic here checks if the jacobian is near singular, which may make the 
    # Jacobian unsafe to invert: It can blow up the computation to massive swings
    # Then if under a threshold (1e-3) we just do an adhoc heuristic moving
    # z in the same direction as the sign of F_z

    # The point is that, in dominance solvable games, the player ALWAYS 
    # Prefers that action, and there is no interior root. 
    # So we use the heuristic to skip the newton step and push towards
    # the same direction as F_z's sign, a safer update, especially for cases 
    # where there is no interior root and we are searchign for pure equilibria. 

    det = np.linalg.det(jacobian)
    newton_delta = np.linalg.solve(jacobian + 1e-4 * np.eye(2), -F_z)
    fallback_delta = 0.1 * np.sign(F_z)
    delta = np.where((np.abs(det) < 1e-3) & (np.linalg.norm(F_z) >= eps), fallback_delta, newton_delta)

    noise_scale = 0.1 / np.sqrt(step_num + 1.0)
    noise = jax.random.normal(key, shape=z.shape) * noise_scale

    z = np.clip(z + delta + noise, 0, 1)
    F_new = compute_F(z, U, is_pt_p1, is_pt_p2, pt, sort_orders)

    p_ok = (np.abs(F_new[0]) < eps) | ((z[0] <= eps) & (F_new[0] < 0)) | ((z[0] >= 1 - eps) & (F_new[0] > 0))
    q_ok = (np.abs(F_new[1]) < eps) | ((z[1] <= eps) & (F_new[1] < 0)) | ((z[1] >= 1 - eps) & (F_new[1] > 0))
    converged = p_ok & q_ok
    #jax.debug.print("z: {}, F: {}, p_ok: {}, q_ok: {}", z, F_new, p_ok, q_ok)

    return z, converged


@functools.partial(jax.jit, static_argnums=(2,3,4))
def compute_F(z, U, is_pt_p1, is_pt_p2, pt, sort_orders):
    U = np.array(U)
    p, q = z
    # Calculate F_i for each player via the indifference equation u(A_1) - u(A_2)
    # Player 1 is rows, player 2 is columns
    row_a_1, row_a_2 = np.array([U[0, 0, 0], U[0, 1, 0]]), np.array([U[1, 0, 0], U[1, 1, 0]])
    col_a_1, col_a_2 = np.array([U[0, 0, 1], U[1, 0, 1]]), np.array([U[0, 1, 1], U[1, 1, 1]])

    # Define the prob vectors for both: p1 is the probs player 2 players action 0 or 1, and vice versa
    p_1_probs, p_2_probs = np.array([q, 1-q]), np.array([p, 1-p])

    if is_pt_p1:
        player_1_F = pt.expected_pt_value(row_a_1, p_1_probs, sort_orders[0]) - pt.expected_pt_value(row_a_2, p_1_probs, sort_orders[1])
    else:
        player_1_F = p_1_probs @ row_a_1 - p_1_probs @ row_a_2 

    if is_pt_p2:
        player_2_F = pt.expected_pt_value(col_a_1, p_2_probs, sort_orders[2]) - pt.expected_pt_value(col_a_2, p_2_probs, sort_orders[3])
    else:
        player_2_F = p_2_probs @ col_a_1 - p_2_probs @ col_a_2
    
    F_z = np.array([player_1_F, player_2_F])
    return F_z

# Deprecated
def compute_jacobian(p, q, util_func, U, p1_type, p2_type, eps=1e-6):
    ''' 
    A numerical computation of the jacobian matrix for the semismooth newton method
    Uses finite difference with the indifference equation
    Do i need to add logic for forward and backward difference near boundaries? 
    I fugured that if we're near a boundary in the first place its a pure strategy response, 
    and the value error already catches it, so i think the only risk is pure/mixed equilibria being missed
    which seems unlikely anyways. 
    '''
    # Check if we are near boundary
    if p + eps > 1 or p - eps < 0 or q + eps > 1 or q - eps < 0:
        raise ValueError(f"probability exceeded threshold: p = {p}, q = {q}, eps = {eps}")

    # Initialize Jacobian Matrix
    J = np.zeros((2, 2))

    # Redefine actions:
    row_a_1, row_a_2 = np.array([U[0, 0, 0], U[0, 1, 0]]), np.array([U[1, 0, 0], U[1, 1, 0]])
    col_a_1, col_a_2 = np.array([U[0, 0, 1], U[1, 0, 1]]), np.array([U[0, 1, 1], U[1, 1, 1]])

    # Compute F1 (only with q, F1 does not depend on p)
    p_1_probs_plus, p_1_probs_minus = np.array([q + eps, 1 - (q + eps)]), np.array([q - eps, 1 - (q - eps)])

    F1_plus = util_func(row_a_1, p_1_probs_plus, p1_type) - util_func(row_a_2, p_1_probs_plus, p1_type)
    F1_minus = util_func(row_a_1, p_1_probs_minus, p1_type) - util_func(row_a_2, p_1_probs_minus, p1_type)
    F1_delta = (F1_plus - F1_minus) / (2 * eps)

    # Compute F2 (only with p, F2 does not depend on F1)
    p_2_probs_plus, p_2_probs_minus = np.array([p + eps, 1 - (p + eps)]), np.array([p - eps, 1 - (p - eps)])

    F2_plus = util_func(col_a_1, p_2_probs_plus, p2_type) - util_func(col_a_2, p_2_probs_plus, p2_type)
    F2_minus = util_func(col_a_1, p_2_probs_minus, p2_type) - util_func(col_a_2, p_2_probs_minus, p2_type)
    F2_delta = (F2_plus - F2_minus) / (2 * eps)

    # Add to Jacobian, 0, 0 and 1, 1 are dF1/dp and dF2/dq, and since F1 only depends on q and F2 only 
    # depends on p, those derivatives will always be zero. 
    J[0, 1], J[1, 0] = F1_delta, F2_delta

    return J

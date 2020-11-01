import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
scale = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
conv_2d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

data = np.random.binomial(1, 0.25, (100000, 1000))
epsilon = np.array([0.5, 0.25, 0.1, 0.01, 0.001])
m = np.arange(1, 1001)
p = 0.25


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.show()


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.show()


def q_11():
    """
    answer for question 11.
    3d plot of a random generated points with multivariate normal distribution
    """
    plot_3d(x_y_z)


def q_12():
    """
    answer for question 12.
    3d plot of a random generated points with multivariate normal distribution
    multiplied by a scaling matrix, and prints the matching covariance matrix
    :return: the data
    """
    s_x_y_z = np.dot(scale, x_y_z)
    plot_3d(s_x_y_z)
    print(np.cov(s_x_y_z))
    return s_x_y_z


def q_13():
    """
    answer for question 13.
    3d plot of a random generated points with multivariate normal distribution
    multiplied by a scaling and orthogonal matrices. The function prints the random
    orthogonal matrix that was used and the matching covariance matrix to the data
    :return: the data
    """
    orthogonal = get_orthogonal_matrix(3)
    print(orthogonal)
    ort_s_x_y_z = np.dot(orthogonal, q_12())
    plot_3d(ort_s_x_y_z)
    print(np.cov(ort_s_x_y_z))
    return ort_s_x_y_z


def q_14():
    """
    answer for question 14.
    2d plot of a random generated points with multivariate normal distribution,
    multiplied by a scaling and orthogonal matrices
    """
    ort_s_x_y_z_2d = np.dot(conv_2d, q_13())
    plot_2d(ort_s_x_y_z_2d)


def q_15():
    """
    answer for question 15.
    2d plot of a random generated points with multivariate normal distribution,
    multiplied by a scaling and orthogonal matrices, and their original z values are
    in the range 0.1>z>-0.4
    """
    ort_s_x_y_z = q_13()
    z_vals = ort_s_x_y_z[2]
    z_smaller_0_1 = z_vals < 0.1
    z_bigger_0_4 = z_vals > -0.4
    wanted_z_val = z_smaller_0_1 * z_bigger_0_4
    x_y_ranged = np.extract([wanted_z_val, wanted_z_val], ort_s_x_y_z).reshape(2, -1)
    plot_2d(x_y_ranged)


def q_16_a():
    """
    answer for question 16-a.
    plot of the data's estimate as a function of m
    """
    data_5_rows = data[:5, :]
    cumsum = np.cumsum(data_5_rows, 1)
    m_5_rows = np.tile(m, (5, 1))
    estimator = cumsum / m_5_rows

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(-0.1, 1.1)
    plt.plot([0, 1000], [0.25, 0.25])
    plt.title('Estimator 5 sequences')
    for i in range(5):
        ax.plot(estimator[i], label="seq " + str(i + 1))
    ax.legend(['0.25', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5'])
    ax.set_xlabel('m')
    ax.set_ylabel('estimator')
    fig.show()


def q_16_b():
    """
    answer for question 16-b,c.
    plots of the upper bounds of chebyshev and hoeffding bounds for each epsilon, and
    the percentage of sequence that satisfy the wanted equation
    """
    # calculations for q_16_b:
    m_5_rows = np.tile(m, (5, 1))
    epsilonT = (epsilon * epsilon).reshape(-1, 1)
    chebyshev_bound = 1 / (4 * m_5_rows * epsilonT)
    hoeffding_bound = 2 * np.exp(-2 * m_5_rows * epsilonT)
    chebyshev_bound = np.where(chebyshev_bound > 1, 1, chebyshev_bound)
    hoeffding_bound = np.where(hoeffding_bound > 1, 1, hoeffding_bound)

    # calculations for q_16_c:
    cumsum = np.cumsum(data, 1)
    m_100000_rows = np.tile(m, (100000, 1))
    estimator = cumsum / m_100000_rows
    estimator -= p
    estimator = np.abs(estimator)

    for i in range(5):
        estimator_bool = estimator >= epsilon[i]
        sums = np.sum(estimator_bool, axis=0)
        sums = sums / 100000

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 1.5)
        plt.title('Chebyshev & Hoeffding bounds, epsilon=' + str(epsilon[i]))
        ax.plot(chebyshev_bound[i])
        ax.plot(hoeffding_bound[i])

        # the next line is the plot for q_16_c:
        ax.plot(sums)

        ax.legend(["chebyshev bound", "hoeffding bound"])
        ax.set_xlabel('m')
        ax.set_ylabel('bound')
        fig.show()


def write_output(ne:int, neq:int, D, RF, le, ls, N):
    """
    write outputs in a .csv file

    input:

    """
    with open('output.txt', 'w') as rpt:
        rpt.write('node, D, RF,e, le, ls, N\n')
        ii = 0
        while ii <= ne - 1:
            rpt.write('{},{:.3f},{:.3f},{},{:.3f},{:.3f},{:.3f}\n'.format(ii + 1, D[ii, 0], RF[ii, 0], ii + 1, le[ii, 0], ls[ii, 0], N[ii, 0]))
            ii = ii + 1
        for ii in range(ne, neq):
            rpt.write('{},{:.3f},{:.3f}\n'.format(ii + 1, D[ii, 0], RF[ii, 0]))


def show_output(ne:int, neq:int, D, RF, le, ls, N):
    """
    write outputs in a .csv file

    input:
    ne : number of elements
    neq : degree of freedom
    D : global displacement vector
    RF : residual vector (reaction forces at supports)
    le : element strain
    ls : element stress
    N : element force

    """
    print('node {:<10} D {:<10} RF \n'.format(' ', ' '))
    for ii in range(neq):
        print('{} {:>15.3f} {:>15.3f} \n'.format(ii + 1, D[ii, 0], RF[ii, 0]))

    print('element {:<3} strain {:<7} stress {:<8} element force\n'.format(' ', ' ', ' ', ' '))
    for ii in range(ne):
        print('{} {:>15.3f} {:>15.3f} {:>15.3f} \n'.format(ii + 1, le[ii, 0], ls[ii, 0], N[ii, 0]))
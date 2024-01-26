import torch



def print_grid(data, data_type=None):
    if data_type == 'sparse':
        flat_grid = torch.cat((torch.full((81, 1), 0.5), data.reshape(81, 9)), dim=1).argmax(dim=1).numpy().astype(int)

    elif data_type == 'dense':
        flat_grid = (data+1).numpy().astype(int)

    else:
        flat_grid = data

    S = ''
    for i in range(9):
        if i % 3 == 0:
            S += '-------------------------------\n'

        for j in range(9):
            if j % 3 == 0:
                S += '|'
            S += f' {str(flat_grid[i*9 + j])} '

        S += '|\n'
            
    S += '-------------------------------'

    print(S)


def decoder(pred):
    pred = pred.argmax(dim=1).view(81, 10)
    pred = pred.argmax(dim=2)
    return pred



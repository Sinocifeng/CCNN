import matplotlib.pyplot as plt


def draw_losses(losses, title):

    print(f" losses is { losses } ,length is {len(losses)}")
    plt.figure() 

    # loss_values = [min(float(val), 100) for val in losses]
    epochs = list(range(1, len(losses) + 1))  

    plt.plot(epochs, losses, marker='.')  
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')   
    plt.title(title)  
    plt.grid(True) 
    plt.show()
    plt.close()

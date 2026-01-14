import numpy as np
from losses import cross_loss 



def train(model, optimizer, x_train, y_train,
          epochs=5, batch_size=32):
    
    n = x_train.shape[0]

    for epoch in range(epochs):
        # Shuffle data every epoch
        indices = np.random.permutation(n)
        x_train = x_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        toatl_acc =0

        for i in range(0, n, batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # 1️⃣ Forward pass
            y_pred = model.forward(x_batch)

            # 2️⃣ Loss
            loss = cross_loss(y_pred, y_batch)
            toatl_acc = model.evaluate(y_pred,y_batch)
            epoch_loss += loss

            # 3️⃣ Backward pass
            grads = model.backward(y_batch)

            # 4️⃣ Update parameters
            optimizer.update(model, grads)

        avg_loss = epoch_loss / (n // batch_size)
        avg_acc = toatl_acc/ (n//batch_size)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc :{avg_acc:.4f}")


/**************************/
/*  Utility               */
/**************************/

var Utility = {

    // FindObject by name
    FindByName: function (Name, Object)
    {
        var element = null;
        if (typeof Object != "object") throw "object must be typeof object";
        for (var i = 0; i < Object.length; i++)
        {
            if (Object[i].hasOwnProperty("_Name") && Object[i]._Name == Name)
            {
                element = Object[i];
                break;
            }
        }
        return element;
    }
};

// This is a simple implementation of the queue data structure (JS doesn't have queues)
class Queue {
    constructor() {
        this.elements = [];
        this.minIndex = 0;
        this.maxIndex = -1;
    }

    Enqueue(element) {
        this.elements.push(element);
        this.maxIndex++;
    }

    Dequeue() {
        if (this.minIndex <= this.maxIndex) {
            const element = this.elements[this.minIndex++];
            return element;
        }
    }

    IsEmpty() {
        return this.minIndex === this.maxIndex + 1;
    }

    Size() {
        return this.maxIndex - this.minIndex + 1;
    }
}

class ContentLoadingQueue {
    constructor(maxNumberOfContentsBeingLoaded) {
        this.numberOfContentsBeingLoaded = 0;
        this.maxNumberOfContentsBeingLoaded = maxNumberOfContentsBeingLoaded;
        this.queue = new Queue();
    }

    QueueTask(task) {
        if (this.numberOfContentsBeingLoaded + 1 > this.maxNumberOfContentsBeingLoaded) {
            this.queue.Enqueue(task);
        } else {
            this._Executetask(task);
        }
    }

    async _Executetask(task) {
        this.numberOfContentsBeingLoaded++;

        await task();

        if (!this.queue.IsEmpty()) {
            const taskFromQueue = this.queue.Dequeue();
            this._Executetask(taskFromQueue);
        }
        this.numberOfContentsBeingLoaded--;
    }
}

window.GlobalContentLoadingQueue = new ContentLoadingQueue(300);
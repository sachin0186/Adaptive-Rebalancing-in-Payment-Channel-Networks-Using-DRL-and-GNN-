import sys


class Transaction:
    def __init__(self, env, topology, time_of_arrival, source, destination, amount, verbose, verbose_also_print_transactions):
        self.env = env
        self.time_of_arrival = time_of_arrival
        self.source = source
        self.destination = destination
        self.amount = amount
        self.verbose = verbose
        self.verbose_also_print_transactions = verbose_also_print_transactions
        self.status = "PENDING"
        self.pathfinder(topology)

        if self.verbose and self.verbose_also_print_transactions:
            print("Time {:.2f}: Transaction {} generated.".format(self.env.now, self))

    def pathfinder(self, topology):
        if self.source == "L" and self.destination == "R":
            self.path = ["L", "N", "R"]
            self.previous_node = "L"
            self.current_node = topology["N"]
            self.next_node = "R"
        elif self.source == "R" and self.destination == "L":
            self.path = ["R", "N", "L"]
            self.previous_node = "R"
            self.current_node = topology["N"]
            self.next_node = "L"
        else:
            print("Input error")
            sys.exit(1)

    def run(self):
        # Check if we can forward the payment
        if self.source == "L" and self.destination == "R":
            # Check if we have enough local balance in the R channel
            if self.current_node.local_balances["R"] >= self.amount:
                # Update balances
                self.current_node.update_balances("L", self.amount, -self.amount)  # Receive on L channel
                self.current_node.update_balances("R", -self.amount, self.amount)  # Forward to R channel
                self.status = "SUCCEEDED"
            else:
                self.status = "FAILED"
        elif self.source == "R" and self.destination == "L":
            # Check if we have enough local balance in the L channel
            if self.current_node.local_balances["L"] >= self.amount:
                # Update balances
                self.current_node.update_balances("R", self.amount, -self.amount)  # Receive on R channel
                self.current_node.update_balances("L", -self.amount, self.amount)  # Forward to L channel
                self.status = "SUCCEEDED"
            else:
                self.status = "FAILED"

        if self.verbose and self.verbose_also_print_transactions:
            print("Time {:.2f}: Transaction {} {}".format(self.env.now, self, self.status))

        yield self.env.timeout(0)  # Yield control back to the simulator

    def get_transaction_signature(self):
        transaction_signature = (self.time_of_arrival, self.source, self.destination, self.amount, self.status)
        return transaction_signature

    def __repr__(self):
        return "%s->%s t=%.2f a=%d" % (self.source, self.destination, self.time_of_arrival, self.amount)

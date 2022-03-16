#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

#define sender 0
#define receiver 1

#define sync_send 0
#define buffered_send 1

void sender_sync_send(long message_size, long repetitions, long delta, long max)
{
	struct timespec tstamp = { 0,0 };
	uint8_t* data = 0;

	struct timespec trecv = { 0,0 };

	printf("Message Size [bytes], Time Taken [s], Time Taken [ms], Time Taken [us], Time Taken [ns]\n");
	for (; message_size <= max; message_size *= delta)
	{
		double s = 0;
		data = calloc(message_size, sizeof(uint8_t));
		for (long _ = 0; _ < repetitions; _++)
		{
			MPI_Barrier(MPI_COMM_WORLD); // sync sender with receiver before send
			timespec_get(&tstamp, TIME_UTC); // closest to real send timestamp
			MPI_Ssend(data, message_size, MPI_BYTE, receiver, 0, MPI_COMM_WORLD);
			MPI_Recv(&trecv, sizeof(tstamp), MPI_BYTE, receiver, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			s += ((double)trecv.tv_sec + 1.0e-9 * trecv.tv_nsec) - ((double)tstamp.tv_sec + 1.0e-9 * tstamp.tv_nsec);
		}
		free(data);
		double avg_s = s / repetitions;
		printf("%d,\t\t%f,\t\t%f,\t\t%f,\t\t%f\n", message_size, avg_s, avg_s * 1e3, avg_s * 1e6, avg_s * 1e9);
	}
}

void sender_buffered_send(long message_size, long repetitions, long delta, long max)
{
	struct timespec tstamp = { 0,0 };
	uint8_t* data = 0;

	struct timespec trecv = { 0,0 };

	printf("Message Size [bytes], Time Taken [s], Time Taken [ms], Time Taken [us], Time Taken [ns]\n");
	for (; message_size <= max; message_size *= delta)
	{
		double s = 0;
		data = calloc(message_size, sizeof(uint8_t));

		int buffer_size;
		MPI_Pack_size(message_size, MPI_BYTE, MPI_COMM_WORLD, &buffer_size);
		buffer_size += MPI_BSEND_OVERHEAD;
		void* buf = malloc(buffer_size);

		MPI_Buffer_attach(buf, buffer_size);

		for (long _ = 0; _ < repetitions; _++)
		{
			MPI_Barrier(MPI_COMM_WORLD); // sync sender with receiver before send
			timespec_get(&tstamp, TIME_UTC); // closest to real send timestamp
			MPI_Bsend(data, message_size, MPI_BYTE, receiver, 0, MPI_COMM_WORLD);
			MPI_Recv(&trecv, sizeof(tstamp), MPI_BYTE, receiver, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			s += ((double)trecv.tv_sec + 1.0e-9 * trecv.tv_nsec) - ((double)tstamp.tv_sec + 1.0e-9 * tstamp.tv_nsec);
		}

		void* bbuf;
		int bsize;
		MPI_Buffer_detach(&bbuf, &bsize);

		free(buf);
		free(data);

		double avg_s = s / repetitions;
		printf("%d,\t\t%f,\t\t%f,\t\t%f,\t\t%f\n", message_size, avg_s, avg_s * 1e3, avg_s * 1e6, avg_s * 1e9);
	}
}

void receiver_recv(long message_size, long repetitions, long delta, long max)
{
	struct timespec tstamp = { 0,0 };
	uint8_t* data = 0;

	for (; message_size <= max; message_size *= delta)
	{
		data = calloc(message_size, sizeof(uint8_t));
		for (long _ = 0; _ < repetitions; _++)
		{
			MPI_Barrier(MPI_COMM_WORLD);// sync sender with receiver before recv
			MPI_Recv(data, message_size, MPI_BYTE, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			timespec_get(&tstamp, TIME_UTC); // closest to real full recv timestamp
			MPI_Ssend(&tstamp, sizeof(tstamp), MPI_BYTE, sender, 0, MPI_COMM_WORLD); // send back recv timestamp
		}
		free(data);
	}
}

typedef void (*sender_f)(long message_size, long repetitions, long delta, long max);

int main(int argc, char* argv[])
{
	if (argc <= 1)
	{
		fprintf(stderr, "Select sender type 0-Ssend, 1-Bsend");
		return -1;
	}

	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size != 2) {
		fprintf(stderr, "World size must be equal to 2 for %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	sender_f sender_routine = atoi(argv[1]) == sync_send ? sender_sync_send : sender_buffered_send;

	long message_size = 1;
	long repetitions = 100000;
	long delta = 2;
	long max = 1; // 512 MB

	if (world_rank == sender) sender_routine(message_size, repetitions, delta, max);
	else if (world_rank == receiver) receiver_recv(message_size, repetitions, delta, max);

	MPI_Finalize();
}
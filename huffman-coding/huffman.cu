#include <stdio.h>
#include <stdlib.h>

#pragma pack(push, 1)
typedef struct {
  unsigned short type;
  unsigned int size;
  unsigned short reserved1;
  unsigned short reserved2;
  unsigned int offset;
  unsigned int width;
  unsigned int height;
} BMP_FILE_HEADER;

typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} BMP_RGB;

#pragma pack(pop)

void read_file(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    printf("Error opening file\n");
    exit(1);
  }

  BMP_FILE_HEADER header;
  fread(&header, sizeof(BMP_FILE_HEADER), 1, fp);

  if (header.type != 0x4D42) { // 'BM' in hexadecimal is 0x4D42
    printf("Error: Not a valid BMP file.\n");
    fclose(fp);
    exit(1);
  }

  int width = header.width;
  int height = header.height;

  BMP_RGB *pixels = (BMP_RGB *)malloc(width * height * sizeof(BMP_RGB));
  if (!pixels) {
    printf("Error allocating memory for pixels\n");
    fclose(fp);
    exit(1);
  }

  fseek(fp, header.offset, SEEK_SET);
  fread(pixels, sizeof(BMP_RGB), width * height, fp);
  fclose(fp);
  free(pixels);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <filename>\n", argv[0]);
    exit(1);
  }
  read_file(argv[1]);
  return 0;
}
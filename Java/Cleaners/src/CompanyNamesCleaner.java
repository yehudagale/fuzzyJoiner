import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

public class CompanyNamesCleaner {

	public static void main(String[] args) throws IOException {
		for(String file : args) {
			BufferedReader r = new BufferedReader(new FileReader(file));
			String line;
			lines: while ((line = r.readLine()) != null) {
				String[] nameList = line.split("\\|");
				Set<String> names = new LinkedHashSet<>();
				Arrays.asList(nameList).forEach((String s) -> { 
					names.add(s.trim()); 
				});
				
				// filter singletons
				if (names.size() <= 1) {
					//System.err.println(line);
					continue;
				}
				
				// filter anything not in ISO-88589-1
					CharsetEncoder x = StandardCharsets.ISO_8859_1.newEncoder();
					for(String s : names) {
						try {
							x.encode(CharBuffer.wrap(s));
						} catch (CharacterCodingException e) {
							continue lines;
						}
					}

				System.out.println(names.stream().reduce((String lh, String rh) -> { return lh + "|" + rh; }).get());
			}
			r.close();
		}
	}
}
